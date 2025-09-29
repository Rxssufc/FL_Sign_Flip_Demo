# simulate.py
import os
import numpy as np
import torch
import flwr as fl

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.simulation import start_simulation

# Reuse client and data loader
from Client import AudioClient  # relies on if __name__ == "__main__" guard in file
from dataset_loader import load_client_dataset

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# -------------------- Config --------------------
NUM_CLIENTS = 10            # change accordingly, make sure each client has a folder in "data" and a match .csv file
NUM_ROUNDS = 5              # change accordingly
SAVE_DIR = "colab_test"     # name model for save after training

# Example: let only 1 client use the GPU at a time. - adjustable - depends on specs of device
USE_GPU = torch.cuda.is_available()
CONCURRENT_CLIENTS = 1 if USE_GPU else 2
GPU_FRACTION = 1.0 / CONCURRENT_CLIENTS if USE_GPU else 0.0

# Optional: reduce CPU thrash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ------------------------------------------------


def _initial_parameters_from_hf():
    """Build initial global params from the HF model (fp16 to keep comms small)."""
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    nds = [p.detach().cpu().numpy().astype(np.float16) for p in model.parameters()]
    return ndarrays_to_parameters(nds)


def _get_client_id(client_proxy):
    for attr in ("cid", "id", "client_id", "node_id"):
        if hasattr(client_proxy, attr):
            return str(getattr(client_proxy, attr))
    return str(client_proxy)


class CaptureFedAvg(fl.server.strategy.FedAvg):
    """Your original strategy, kept as-is, with minor tweaks for simulation."""
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"val_round": rnd},
            initial_parameters=_initial_parameters_from_hf(),
        )
        self.final_parameters = None
        self.latest_weights = None

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
        client_updates_f32 = []
        for _, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            arrs_f32 = [a.astype(np.float32, copy=False) for a in arrs]
            client_updates_f32.append(arrs_f32)

        n_clients = len(client_updates_f32)
        n_tensors = len(client_updates_f32[0])
        avg_weights_f32 = []
        for i in range(n_tensors):
            stacked = np.stack([client_updates_f32[c][i] for c in range(n_clients)], axis=0)
            avg = stacked.mean(axis=0)
            avg_weights_f32.append(avg)

        # Keep fp16 for broadcast
        avg_weights_f16 = [w.astype(np.float16) for w in avg_weights_f32]
        self.latest_weights = avg_weights_f16
        self.final_parameters = ndarrays_to_parameters(avg_weights_f16)

        # Aggregate/print train loss if provided
        losses = [fit_res.metrics.get("train_loss") for _, fit_res in results if "train_loss" in fit_res.metrics]
        if losses:
            print(f"[Round {rnd}] Avg Train Loss: {sum(losses)/len(losses):.4f}")
        else:
            print(f"[Round {rnd}] No train_loss reported")

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
    weights_f32 = [w.astype(np.float32, copy=False) for w in weights]

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    state_keys = list(model.state_dict().keys())

    state_dict = {k: torch.tensor(w) for k, w in zip(state_keys, weights_f32)}
    model.load_state_dict(state_dict, strict=False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h").save_pretrained(SAVE_DIR)
    print(f" Model and processor saved to '{SAVE_DIR}'")


def client_fn(cid: str) -> fl.client.Client:
    """
    Factory for virtual clients.
    `cid` is a string flower assigns: "0", "1", ..., str(NUM_CLIENTS-1) by default.
    We'll map that to your 1-based client IDs/files.
    """
    # Map "0"->1, "1"->2, ... to match csv naming
    client_id = int(cid) + 1

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # IMPORTANT: batch_size=1 for memory safety, same as client.py
    train_loader, test_loader = load_client_dataset(client_id, processor, adversarial=False, batch_size=1)

    client = AudioClient(client_id, model, processor, train_loader, test_loader)
    # AudioClient is a NumPyClient; convert to Client for simulation
    return client.to_client()


def main():
    # Strategy with initial params + your custom aggregation
    strategy = CaptureFedAvg()

    # Control resource usage per (virtual) client
    # - On GPU: run only CONCURRENT_CLIENTS at a time; each gets 1/CONCURRENT_CLIENTS of the GPU.
    # - On CPU-only: allow a couple in parallel.
    client_resources = {
        "num_cpus": 1,
        "num_gpus": GPU_FRACTION if USE_GPU else 0.0,
    }

    print(f"🏁 Starting simulation: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds")
    print(f"   Resources per client: {client_resources} (Colab GPU available: {USE_GPU})")

    # Run simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=None),
        strategy=strategy,
        client_resources=client_resources,
        # actors optional in recent flwr, keeping defaults keeps it simple
    )

    # Save final model
    if strategy.final_parameters is not None:
        save_model(strategy.final_parameters)
    else:
        print(" No final parameters captured; nothing to save.")


if __name__ == "__main__":
    main()
