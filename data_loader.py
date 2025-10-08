"""
data_loader.py — Prep utility for AlpaCare Medical Instruction Assistant

- Loads `lavita/AlpaCare-MedInstruct-52k` from Hugging Face
- Standardizes to {prompt, response}
- Splits into train/val/test (90/5/5)
- Optional caps for quick Colab runs
- Saves JSONL files under data/processed/ + a small metadata.json

Run:
  python data_loader.py
  # or with caps:
  python data_loader.py --max-train 5000 --max-val 500 --max-test 500
"""

from dataclasses import dataclass
from typing import Optional, Dict
import os, json, argparse
from datasets import load_dataset, Dataset, DatasetDict


# Config


@dataclass
class SplitConfig:
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    max_train: Optional[int] = 10000
    max_val: Optional[int] = 500
    max_test: Optional[int] = 500
    num_proc: int = 1   # set to >1 on CPUs to speed up map()


# Helpers


def _norm(s: str) -> str:
    # basic whitespace normalization
    return " ".join((s or "").replace("\r", " ").replace("\t", " ").split())

def _standardize_record(rec: Dict) -> Optional[Dict]:
    """
    Standardize raw record to:
      { "prompt": "Instruction: <...>\nInput: <...>\n\nResponse:", "response": "<...>" }
    """
    instr_keys = ["instruction", "question", "prompt", "input_instruction"]
    input_keys = ["input", "context", "additional_input"]
    out_keys   = ["output", "answer", "response", "target"]

    instruction = next((rec.get(k) for k in instr_keys if rec.get(k)), None)
    user_input  = next((rec.get(k) for k in input_keys if rec.get(k)), None)
    output      = next((rec.get(k) for k in out_keys   if rec.get(k)), None)

    if not instruction and not user_input:
        return None
    if not output:
        return None

    instruction = _norm(instruction or "")
    user_input  = _norm(user_input or "")
    output      = _norm(output or "")

    if user_input:
        prompt = f"Instruction: {instruction}\nInput: {user_input}\n\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\n\nResponse:"

    # very short/noisy lines guard
    if len(prompt) < 20 or len(output) < 5:
        return None

    return {"prompt": prompt, "response": output}

def _dataset_to_standardized(ds: Dataset, num_proc: int) -> Dataset:
    mapped = ds.map(lambda r: _standardize_record(r),
                    num_proc=num_proc, remove_columns=ds.column_names)
    mapped = mapped.filter(lambda r: r["prompt"] is not None and r["response"] is not None)
    return mapped

def _cap_split(ds: Dataset, cap: Optional[int]) -> Dataset:
    if cap is None:
        return ds
    return ds.select(range(min(len(ds), cap)))

def _json_escape(x: str) -> str:
    return '"' + x.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'

def save_jsonl(ds: Dataset, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in ds:
            f.write(f'{{"prompt": {_json_escape(rec["prompt"])}, "response": {_json_escape(rec["response"])}}}\n')

def save_metadata(meta: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# Public API


def load_and_prepare(cfg: SplitConfig = SplitConfig()) -> DatasetDict:
    raw = load_dataset("lavita/AlpaCare-MedInstruct-52k")
    base = _dataset_to_standardized(raw["train"], cfg.num_proc)

    # Split: first test, then val from remaining, to get ~90/5/5
    dsd = base.train_test_split(test_size=cfg.test_ratio, seed=cfg.seed)
    train_full, test = dsd["train"], dsd["test"]

    val_size = cfg.val_ratio / (1.0 - cfg.test_ratio)  # fraction of remaining for val
    dsd_tv = train_full.train_test_split(test_size=val_size, seed=cfg.seed)
    train, val = dsd_tv["train"], dsd_tv["test"]

    # Optional caps
    train = _cap_split(train, cfg.max_train)
    val   = _cap_split(val,   cfg.max_val)
    test  = _cap_split(test,  cfg.max_test)

    return DatasetDict(train=train, validation=val, test=test)

def save_splits(dsd: DatasetDict, root: str = "data/processed"):
    paths = {
        "train": os.path.join(root, "train.jsonl"),
        "validation": os.path.join(root, "validation.jsonl"),
        "test": os.path.join(root, "test.jsonl"),
    }
    save_jsonl(dsd["train"], paths["train"])
    save_jsonl(dsd["validation"], paths["validation"])
    save_jsonl(dsd["test"], paths["test"])

    meta = {
        "counts": {
            "train": len(dsd["train"]),
            "validation": len(dsd["validation"]),
            "test": len(dsd["test"]),
        },
        "config": {
            "val_ratio": 0.05, "test_ratio": 0.05, "seed": 42
        },
        "paths": paths
    }
    save_metadata(meta, os.path.join(root, "metadata.json"))
    return paths

# CLI


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-val", type=int, default=500)
    p.add_argument("--max-test", type=int, default=500)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/processed")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    cfg = SplitConfig(
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        num_proc=args.num_proc
    )
    dsd = load_and_prepare(cfg)
    out_paths = save_splits(dsd, args.out)
    print("Saved splits:")
    for k, v in out_paths.items():
        print(f"  {k}: {v}")
