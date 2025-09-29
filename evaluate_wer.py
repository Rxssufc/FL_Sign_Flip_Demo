import os
import pandas as pd
import torch
import torchaudio
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, compute_measures
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# === Config ===
CSV_PATH   = "./data/test_samples.csv"  # CSV with 'filename','text'
MODEL_DIR  = "Demo_10%_Adv_100xFull_Inversion"
TARGET_SR  = 16000
OUT_CSV    = "Demo_10%_Adv_100xFull_Inversion.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model & processor ===
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

# === Load test data ===
df = pd.read_csv(CSV_PATH)

# === Resampling helper ===
resamplers = {}
def load_mono_resampled(path: str):
    waveform, sr = torchaudio.load(path)
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resamplers[sr](waveform)
        sr = TARGET_SR
    return waveform.squeeze(0).to(torch.float32), sr

# === Text normalizer ===
normalizer = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

# === Run evaluation ===
results = []
wers = []

print("\n=== Evaluating WER per clip ===")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    audio_path = str(row["filename"])
    ref_raw = str(row["text"])

    try:
        wav, sr = load_mono_resampled(audio_path)
        inputs = processor(
            wav.numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyp_raw = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f" Error processing {audio_path}: {e}")
        hyp_raw = ""

    # Normalize
    ref = normalizer(ref_raw)
    hyp = normalizer(hyp_raw)

    # Compute measures
    m = compute_measures(ref, hyp)
    sample_wer = m["wer"]

    wers.append(sample_wer)

    # === print per sample ===
    print("\n-----------------------------")
    print(f" File: {os.path.basename(audio_path)}")
    print(f"   REF: {ref_raw}")
    print(f"   HYP: {hyp_raw}")
    print(f"   [Normalized] REF='{ref}' | HYP='{hyp}'")
    print(f"    WER: {sample_wer:.2%} | Subs: {m['substitutions']} | Dels: {m['deletions']} | Ins: {m['insertions']} | Ref words: {m['truth']}")

    # Save structured results
    results.append({
        "filename": audio_path,
        "reference_raw": ref_raw,
        "hypothesis_raw": hyp_raw,
        "reference_norm": ref,
        "hypothesis_norm": hyp,
        "wer": sample_wer,
        "subs": m["substitutions"],
        "dels": m["deletions"],
        "ins": m["insertions"],
        "ref_words": m["truth"]
    })

# === Save results to CSV ===
pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"\n✅ Detailed results saved to {OUT_CSV}")

# === Print average WER ===
avg_wer = sum(wers) / len(wers) if wers else 0.0
print(f"\n=== Overall WER across {len(wers)} clips: {avg_wer:.2%} ===")
