# client.py
import argparse
import random
import numpy as np
import torch
import flwr as fl
from torch.amp import autocast
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
from dataset_loader import load_client_dataset

# -------- Configuration --------
inversion_strength = 100.0   # strength of inversion attack
ADVERSARIAL_CLIENTS = [1]    # list of adversarial client IDs
USE_AMP = True               # mixed precision to cut VRAM if CUDA is available
# -------------------------------


class AudioClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, processor, train_loader, test_loader):
        self.client_id = int(client_id)

        # Stay on CPU in __init__; move to CUDA only inside fit/evaluate
        self.device = torch.device("cpu")
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Seed reproducibility
        seed = 1234 + self.client_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Freeze layers - leaves 20% trainable - time saving
        if hasattr(self.model, "freeze_feature_encoder"):
            self.model.freeze_feature_encoder()
        try:
            for i in range(10):
                for param in self.model.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False
        except Exception:
            pass

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Client {self.client_id} started (seed={seed})")
        print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")
        if self.client_id in ADVERSARIAL_CLIENTS:
            print(f" Client {self.client_id} is ADVERSARIAL (inversion_strength={inversion_strength})")

        # AMP scaler (effective only on CUDA)
        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())

    # --- helpers -------------------------------------------------------------

    def _enter_device(self):
        """Move model to active device (CUDA if available), return that device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.model.to(device)
        return device

    def _leave_device(self):
        """Move model back to CPU and clear CUDA cache to avoid idle VRAM usage."""
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cleanup_tensors(self, *tensors):
        for t in tensors:
            try:
                del t
            except Exception:
                pass

    # ------------------------------------------------------------------------

    def get_parameters(self, config):
        # Send weights in float16 to reduce memory/comm size
        return [p.detach().cpu().numpy().astype(np.float16) for p in self.model.parameters()]

    def set_parameters(self, parameters):
        # Incoming may be float16; cast to model param dtype (usually float32)
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p, dtype=p.dtype, device=p.device)

    def fit(self, parameters, config):
        # Copy incoming (could be fp16); keep a float16 copy for poisoned math later
        global_params_np_f16 = [np.copy(arr).astype(np.float16, copy=False) for arr in parameters]

        # Load into model (cast to model dtype inside set_parameters)
        self.set_parameters(parameters)

        device = self._enter_device()
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        total_loss = 0.0
        amp_enabled = USE_AMP and torch.cuda.is_available()

        for batch in tqdm(self.train_loader, desc=f"Client {self.client_id} Training", leave=True):
            iv = batch["input_values"].to(device, non_blocking=False)
            am = batch["attention_mask"].to(device, non_blocking=False)
            lbl = batch["labels"].to(device, non_blocking=False)
            tgt_len = batch["label_lengths"].to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = self.model(iv, attention_mask=am).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                batch_size, T_out, _ = logits.size()
                inp_len = torch.full((batch_size,), T_out, dtype=torch.long, device=device)

                loss = torch.nn.functional.ctc_loss(
                    log_probs.transpose(0, 1),
                    lbl,
                    inp_len,
                    tgt_len,
                    blank=self.processor.tokenizer.pad_token_id,
                    zero_infinity=True,
                )

            if amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Free per-batch tensors
            self._cleanup_tensors(iv, am, lbl, tgt_len, logits, log_probs, inp_len)

        avg_loss = total_loss / max(1, len(self.train_loader))

        # Extract local params and send as fp16
        local_params_np_f16 = [p.detach().cpu().numpy().astype(np.float16) for p in self.model.parameters()]

        # Leave device and clear cache
        self._leave_device()

        # Return poisoned or honest weights
        if self.client_id in ADVERSARIAL_CLIENTS:
            alpha = float(inversion_strength)
            # Do inversion math in float32, then cast to float16 for sending
            poisoned_f32 = [
                g.astype(np.float32) - alpha * (l.astype(np.float32) - g.astype(np.float32))
                for g, l in zip(global_params_np_f16, local_params_np_f16)
            ]
            poisoned_f16 = [p.astype(np.float16) for p in poisoned_f32]
            return poisoned_f16, len(self.train_loader.dataset), {
                "train_loss": avg_loss,
                "adversarial": 1,
                "inversion_strength": alpha,
            }
        else:
            return local_params_np_f16, len(self.train_loader.dataset), {
                "train_loss": avg_loss,
                "adversarial": 0,
                "inversion_strength": 0.0,
            }

    def evaluate(self, parameters, config):
        # Parameters may be fp16; set_parameters casts to model dtype
        self.set_parameters(parameters)
        device = self._enter_device()
        self.model.eval()

        total_loss = 0.0
        amp_enabled = USE_AMP and torch.cuda.is_available()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Client {self.client_id} Eval", leave=True):
                iv = batch["input_values"].to(device, non_blocking=False)
                am = batch["attention_mask"].to(device, non_blocking=False)
                lbl = batch["labels"].to(device, non_blocking=False)
                tgt_len = batch["label_lengths"].to(device, non_blocking=False)

                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = self.model(iv, attention_mask=am).logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    batch_size, T_out, _ = logits.size()
                    inp_len = torch.full((batch_size,), T_out, dtype=torch.long, device=device)

                    loss = torch.nn.functional.ctc_loss(
                        log_probs.transpose(0, 1),
                        lbl,
                        inp_len,
                        tgt_len,
                        blank=self.processor.tokenizer.pad_token_id,
                        zero_infinity=True,
                    )
                total_loss += loss.item()

                # Free per-batch tensors
                self._cleanup_tensors(iv, am, lbl, tgt_len, logits, log_probs, inp_len)

        avg_eval_loss = total_loss / max(1, len(self.test_loader))

        # Leave device and clear cache
        self._leave_device()

        return avg_eval_loss, len(self.test_loader.dataset), {"eval_loss": avg_eval_loss}


# ----------------- Run Client -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client_id", type=int, help="Numeric ID of the client (e.g. 1,2,3,4,5)")
    args = parser.parse_args()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Keep batch_size=1 for memory safety - more than likely okay for a larger batch size on better devices
    train_loader, test_loader = load_client_dataset(args.client_id, processor, adversarial=False, batch_size=1)

    client = AudioClient(args.client_id, model, processor, train_loader, test_loader)

    # Use non-deprecated start; NumPyClient -> .to_client()
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
