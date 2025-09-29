# dataset_loader.py
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

MAX_SECONDS = 10.0            # cap each clip to 10 s (adjust if you need more)
TARGET_SR = 16000
MAX_SAMPLES = int(TARGET_SR * MAX_SECONDS)

class CaptchaDataset(Dataset):
    def __init__(self, csv_path, processor):
        """
        Args:
            csv_path: CSV with columns ['filename', 'text']
            processor: HuggingFace Wav2Vec2Processor
        """
        self.processor = processor
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        self.samples = df[['filename', 'text']].values.tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, raw_label = self.samples[idx]

        # Load as float32 (torchaudio default) [channels, time]
        waveform, sample_rate = torchaudio.load(wav_path)

        # Convert to mono if multi-channel
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if needed
        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            sample_rate = TARGET_SR

        # Truncate to prevent huge allocations in feature extractor
        if waveform.size(-1) > MAX_SAMPLES:
            waveform = waveform[..., :MAX_SAMPLES]

        # Prepare label
        label = raw_label.strip().upper()

        # Processor expects numpy or list; keep float32
        audio_np = waveform.squeeze(0).contiguous().numpy()


        inputs = self.processor(
            audio_np,
            sampling_rate=sample_rate,
            padding="longest",
            truncation=True,
            max_length=MAX_SAMPLES,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = torch.tensor(
            self.processor.tokenizer(label).input_ids,
            dtype=torch.long,
        )
        label_length = torch.tensor(len(labels), dtype=torch.long)

        return {
            "input_values": inputs.input_values.squeeze(0),     # [T]
            "attention_mask": inputs.attention_mask.squeeze(0), # [T]
            "labels": labels,
            "label_length": label_length,
        }

def load_client_dataset(client_id, processor, adversarial=False, batch_size=1):
    # Prefer an _adv CSV only if it exists; otherwise fall back.
    # _adv can be used to test training clients with label flipping attacks etc.
    adv_csv = f"./data/client_{client_id}_adv.csv"
    base_csv = f"./data/client_{client_id}.csv"

    if adversarial and os.path.exists(adv_csv):
        csv_path = adv_csv
    else:
        csv_path = base_csv
        if adversarial and not os.path.exists(adv_csv):
            print(f"[load_client_dataset] No _adv CSV for client {client_id}; using {base_csv}.")

    dataset = CaptchaDataset(csv_path, processor)
    pad_id = processor.tokenizer.pad_token_id

    def collate_fn(batch):
        iv = [x["input_values"] for x in batch]
        am = [x["attention_mask"] for x in batch]
        lbl = [x["labels"] for x in batch]
        ll  = [x["label_length"] for x in batch]

        iv = torch.nn.utils.rnn.pad_sequence(iv, batch_first=True)
        am = torch.nn.utils.rnn.pad_sequence(am, batch_first=True)
        lbl = torch.nn.utils.rnn.pad_sequence(lbl, batch_first=True, padding_value=pad_id)
        ll  = torch.stack(ll)

        return {
            "input_values": iv,
            "attention_mask": am,
            "labels": lbl,
            "label_lengths": ll,
        }

    # Simple 80/20 split
    n = len(dataset)
    train_size = int(0.8 * n)
    test_size  = n - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # single-process loaders; no persistent workers; small batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, test_loader
