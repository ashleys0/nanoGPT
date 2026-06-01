#!/usr/bin/env python3
"""Evaluate all checkpoints in ckpt/ep4/ on val.bin and print a table."""

import glob
import math
import os
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig



def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state = ckpt["model"]
    # strip possible compile/DDP prefix
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, ckpt.get("iter_num", -1), ckpt.get("best_val_loss", None)


@torch.no_grad()
def compute_metrics(model, data):
    total_loss = 0.0
    total_tokens = 0
    n_chunks = (len(data) - 1) // BLOCK_SIZE
    n_chunks = (n_chunks // BATCH_SIZE) * BATCH_SIZE

    for i in trange(0, n_chunks, BATCH_SIZE):
        x = torch.stack([data[j * BLOCK_SIZE: j * BLOCK_SIZE + BLOCK_SIZE]
                         for j in range(i, i + BATCH_SIZE)]).to(DEVICE)
        y = torch.stack([data[j * BLOCK_SIZE + 1: j * BLOCK_SIZE + BLOCK_SIZE + 1]
                         for j in range(i, i + BATCH_SIZE)]).to(DEVICE)
        _, loss = model(x, targets=y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return {
        "avg_loss_nats": avg_loss,
        "perplexity": math.exp(avg_loss),
        "bits_per_token": avg_loss / math.log(2),
        "tokens": total_tokens,
    }

# quick eval of 1 ckpt on val set
def eval_one_ckpt(ckpt_path):
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
    data = torch.from_numpy(data.astype(np.int64))
    print(f"Evaluating ckpt: {ckpt_path}:")
    model, _, _ = load_model(ckpt_path)
    m = compute_metrics(model, data)
    print(m)


# eval a bunch that it's in a results.tsv file
def main():
    import re
    import csv

    data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
    data = torch.from_numpy(data.astype(np.int64))

    # If no TSV is supplied, evaluate every ckpt in CKPT_DIR (ckpt/ep_{ep_num})
    # and write a single combined results TSV with one row per ckpt.
    if TSV_PATH is None:
        ckpts = sorted(
            glob.glob(os.path.join(CKPT_DIR, "*.pt")),
            key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)", os.path.basename(p))) else -1,
        )
        out_path = os.path.join(CKPT_DIR, f"ep{ep_num}_results.tsv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["ep_num", "ckpt", "val_loss", "perplexity", "loss_nat"])
            for ckpt_path in ckpts:
                print(f"Evaluating {ckpt_path}...")
                model, _, val_loss = load_model(ckpt_path)
                m = compute_metrics(model, data)
                print(f"  {os.path.basename(ckpt_path)}: "
                      f"ppl={m['perplexity']:.4f} loss_nat={m['avg_loss_nats']:.5f}")
                writer.writerow([
                    ep_num,
                    os.path.basename(ckpt_path),
                    f"{float(val_loss):.5f}" if val_loss is not None else "",
                    f"{m['perplexity']:.4f}",
                    f"{m['avg_loss_nats']:.5f}",
                ])
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        print(f"Wrote {out_path}")
        return

    ckpts = sorted(
        glob.glob(os.path.join(CKPT_DIR, "**", "*best.pt"), recursive=True),
        key=lambda p: int(re.search(r"ep(\d+)", p).group(1)),
    )

    ep_metrics = {}
    for ckpt_path in ckpts:
        ep = int(re.search(r"ep(\d+)", ckpt_path).group(1))
        print(f"Evaluating ep{ep}: {ckpt_path}...")
        model, _, val_loss = load_model(ckpt_path)
        m = compute_metrics(model, data)
        ep_metrics[ep] = m
        print(f"  ep{ep}: ppl={m['perplexity']:.4f} loss_nat={m['avg_loss_nats']:.5f}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(TSV_PATH, "r", newline="") as f:
        reader = list(csv.reader(f, delimiter="\t"))
    header = reader[0]
    rows = [r for r in reader[1:] if r]

    if "perplexity" not in header:
        header.append("perplexity")
    if "loss_nat" not in header:
        header.append("loss_nat")
    ppl_idx = header.index("perplexity")
    nat_idx = header.index("loss_nat")
    ep_idx = header.index("ep_num")

    for r in rows:
        while len(r) < len(header):
            r.append("")
        ep = int(r[ep_idx])
        if ep in ep_metrics:
            r[ppl_idx] = f"{ep_metrics[ep]['perplexity']:.4f}"
            r[nat_idx] = f"{ep_metrics[ep]['avg_loss_nats']:.5f}"

    with open(TSV_PATH, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Updated {TSV_PATH}")


if __name__ == "__main__":
    ep_num = 20
    CKPT_DIR = os.path.join(os.path.dirname(__file__), "ckpt", f"ep_{ep_num}")
    DATA_PATH = '/data2/ash/251B/nanogpt1/val.bin'
    DATA_PATH = '../val.bin'
    # TSV_PATH = os.path.join(os.path.dirname(__file__), "results_grid.tsv")
    TSV_PATH = None 
    BLOCK_SIZE = 1024
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main()

    # eval_one_ckpt(f'ckpt/ep_{ep_num}/ep{ep_num}_best.pt')
