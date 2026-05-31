
import os
import json
import matplotlib
matplotlib.use("Agg")  # safe for headless servers (e.g. over SSH)
import matplotlib.pyplot as plt

def log_train_loss(ep_num, iter_num, lossf, out_dir="."):
    json_path = os.path.join(out_dir, f"ep{ep_num}_train_loss.json")
    plot_path = os.path.join(out_dir, f"ep{ep_num}_train_loss_plot.png")

    # 1) load existing data, or initialize if file doesn't exist
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {"iters": [], "losses": []}

    # 2) append the new loss
    data["iters"].append(int(iter_num))
    data["losses"].append(float(lossf))
    with open(json_path, "w") as f:
        json.dump(data, f)

    # 3) overwrite the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data["iters"], data["losses"], linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("train loss")
    ax.set_title(f"epoch {ep_num} train loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)  # close to avoid leaking figures across many calls


# ---- print architecture and parameter breakdown ----
def write_param_report(model, path='num_params.txt'):
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL ARCHITECTURE")
    lines.append("=" * 80)
    lines.append(str(model))
    lines.append("")
    lines.append("=" * 80)
    lines.append("PARAMETER BREAKDOWN BY MODULE")
    lines.append("=" * 80)
    lines.append(f"{'Module':<60} {'Shape':<25} {'Params':>15}")
    lines.append("-" * 100)

    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        shape_str = str(tuple(p.shape))
        lines.append(f"{name:<60} {shape_str:<25} {n:>15,}")

    lines.append("-" * 100)
    lines.append("")
    lines.append("=" * 80)
    lines.append("PARAMETER GROUPS (high-level)")
    lines.append("=" * 80)

    # group by top-level submodule for a cleaner summary
    from collections import defaultdict
    groups = defaultdict(int)
    for name, p in model.named_parameters():
        top = name.split('.')[0:2]  # e.g. "transformer.wte"
        key = '.'.join(top)
        groups[key] += p.numel()
    for key, n in sorted(groups.items(), key=lambda x: -x[1]):
        pct = 100 * n / total
        lines.append(f"{key:<40} {n:>15,}  ({pct:5.2f}%)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("TOTALS")
    lines.append("=" * 80)
    lines.append(f"Total params:     {total:>15,}  ({total/1e6:.2f}M)")
    lines.append(f"Trainable params: {trainable:>15,}  ({trainable/1e6:.2f}M)")
    lines.append(f"Frozen params:    {total - trainable:>15,}")

    print(f"Total params:     {total:>15,}  ({total/1e6:.2f}M)")
    print(f"Trainable params: {trainable:>15,}  ({trainable/1e6:.2f}M)")
    print(f"Frozen params:    {total - trainable:>15,}")

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n[wrote param report to {path}]")

