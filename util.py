

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

