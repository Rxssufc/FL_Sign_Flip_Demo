# server.py
import os
import flwr as fl
import torch
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAVE_DIR = "Demo_10%_Adv_100xFull_Inversion"  # directory name

def _get_client_id(client_proxy):
    for attr in ("cid", "id", "client_id", "node_id"):
        if hasattr(client_proxy, attr):
            return str(getattr(client_proxy, attr))
    return str(client_proxy)

class CaptureFedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=10,
            min_available_clients=10,
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"val_round": rnd},
        )
        self.final_parameters = None
        self.latest_weights = None  # store the last global weights (fp16) for convenience

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Log per-client meta
        for client_proxy, fit_res in results:
            client_id = _get_client_id(client_proxy)
            metrics = fit_res.metrics or {}
            adv_flag = int(metrics.get("adversarial", 0))
            inv_strength = float(metrics.get("inversion_strength", 0.0))
            if adv_flag == 1:
                print(f"[Round {rnd}] Received Client {client_id} — Inverted weights (strength={inv_strength:.2f})")
            else:
                print(f"[Round {rnd}] Received Client {client_id} — Honest weights")

        # Collect client updates as float32 for stable averaging
        # (Clients might send float16 to reduce memory; cast here.)
        client_updates_f32 = []
        for _, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            arrs_f32 = [a.astype(np.float32, copy=False) for a in arrs]
            client_updates_f32.append(arrs_f32)

        #  Unweighted average (clients all have ~equal num_examples)
        #  true FedAVg uses weighted average, however all clients share same no. samples in this setup
        n_clients = len(client_updates_f32)
        n_tensors = len(client_updates_f32[0])
        avg_weights_f32 = []
        for i in range(n_tensors):
            stacked = np.stack([client_updates_f32[c][i] for c in range(n_clients)], axis=0)
            avg = stacked.mean(axis=0)
            avg_weights_f32.append(avg)

        # Keep a float16 copy to reduce memory/comms back to clients
        avg_weights_f16 = [w.astype(np.float16) for w in avg_weights_f32]
        self.latest_weights = avg_weights_f16
        self.final_parameters = ndarrays_to_parameters(avg_weights_f16)

        # Aggregate/print train loss if provided
        losses = [fit_res.metrics.get("train_loss") for _, fit_res in results if "train_loss" in fit_res.metrics]
        if losses:
            print(f"[Round {rnd}] Avg Train Loss: {sum(losses)/len(losses):.4f}")
        else:
            print(f"[Round {rnd}] No train_loss reported")

        # Return fp16 global parameters to clients (smaller message)
        return self.final_parameters, {}

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated, _ = super().aggregate_evaluate(rnd, results, failures)
        if results:
            losses = [eval_res.metrics.get("eval_loss") for _, eval_res in results if "eval_loss" in eval_res.metrics]
            if losses:
                print(f"[Round {rnd}] Avg Eval Loss: {sum(losses)/len(losses):.4f}")
        return aggregated, {}

def save_model(parameters: fl.common.Parameters):
    print("\n Saving final model and processor...")
    weights = parameters_to_ndarrays(parameters)

    # Cast arrays back to float32 before loading into the model state_dict
    weights_f32 = [w.astype(np.float32, copy=False) for w in weights]

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    state_keys = list(model.state_dict().keys())

    # Align by order
    state_dict = {k: torch.tensor(w) for k, w in zip(state_keys, weights_f32)}
    model.load_state_dict(state_dict, strict=False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h").save_pretrained(SAVE_DIR)
    print(f" Model and processor saved to '{SAVE_DIR}'")

if __name__ == "__main__":
    strategy = CaptureFedAvg()

    # Build a Server so we can serialize per-client work within a round
    server = fl.server.Server(
        client_manager=fl.server.client_manager.SimpleClientManager(),
        strategy=strategy,
    )
    server.set_max_workers(1)  # process selected clients one-by-one

    # Longer timeout since rounds take longer when sequential
    cfg = fl.server.ServerConfig(num_rounds=5, round_timeout=None)

    fl.server.start_server(
        server_address="127.0.0.1:8080",  # prefer IPv4 on Windows
        server=server,
        config=cfg,
    )

    if strategy.final_parameters is not None:
        save_model(strategy.final_parameters)
