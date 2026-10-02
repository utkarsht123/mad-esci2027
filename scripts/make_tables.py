"""Turn results/ into LaTeX tables, number macros and figures under paper/generated/.

Every number in the paper comes from this script, so the text, tables and result files cannot drift apart.
Missing results are rendered as "--" so the paper still compiles before all runs are done.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAMES = {
    "rgb": "RGB",
    "gray": "Grayscale",
    "ir": "Thermal (LWIR)",
    "rgbt": "Early fusion (RGB-T, paired)",
    "rgbt_unpaired": "Early fusion (RGB-T, unpaired)",
    "late_wbf": "Late fusion (WBF)",
    "late_nms": "Late fusion (NMS)",
}
ORDER = list(NAMES)
CH = {"rgb": "3", "gray": "3$^\\dagger$", "ir": "3$^\\dagger$", "rgbt": "4", "rgbt_unpaired": "4", "late_wbf": "3+3", "late_nms": "3+3"}
ROBUST = ["clean", "rgb_lowlight", "rgb_fog", "rgb_noise", "rgb_blackout", "ir_noise", "ir_blackout"]
ROBUST_HDR = ["Clean", "Low-light", "Fog", "Noise", "Off", "Noise", "Off"]
SHIFTS = [0, 4, 8, 16, 32]
# must match the comparisons bootstrapped in scripts/evaluate.py
DIFF_PAIRS = [("rgb", "gray"), ("ir", "rgb"), ("rgbt", "ir"), ("rgbt", "rgb"), ("rgbt", "rgbt_unpaired"), ("rgbt", "late_wbf"), ("late_wbf", "ir")]


def pct(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def ms(sub: pd.DataFrame, key: str):
    """mean (and std if >1 seed) in percent."""
    if sub.empty:
        return "--", None
    v = sub[key].to_numpy()
    if len(v) > 1:
        return f"{100 * v.mean():.1f}{{\\scriptsize$\\pm${100 * v.std(ddof=1):.1f}}}", v.mean()
    return f"{100 * v.mean():.1f}", v.mean()


NUMWORDS = {"0": "Zero", "4": "Four", "8": "Eight", "16": "Sixteen", "32": "ThirtyTwo"}


def macro_name(*parts):
    r"""LaTeX-safe macro name (letters only), e.g. ("rgbt", "ir_shift16", "ApFifty") -> \RRgbtIrShiftSixteenApFifty."""
    txt = re.sub(r"\d+", lambda m: "_" + NUMWORDS[m.group()] + "_", "_".join(parts))
    s = "".join(p[0].upper() + p[1:] for p in re.split(r"[^A-Za-z]+", txt) if p)
    return "\\R" + s  # e.g. \RRgbtCleanAllApFifty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    macros = {}

    mfile = a.results / "metrics.csv"
    df = pd.read_csv(mfile) if mfile.exists() else pd.DataFrame(columns=["model", "seed", "condition", "subset", "AP", "AP50", "AP75", "R100"])
    models = [m for m in ORDER if m in set(df.model)] or ORDER
    seeds = sorted(set(df.seed)) if len(df) else []
    boot = json.loads((a.results / "bootstrap.json").read_text()) if (a.results / "bootstrap.json").exists() else {}
    subsets = json.loads((a.results / "subsets.json").read_text()) if (a.results / "subsets.json").exists() else {}
    macros["\\NSeeds"] = str(len(seeds)) if seeds else "--"
    macros["\\NTest"] = str(subsets.get("n_test", "--"))
    macros["\\NTestPersons"] = str(subsets.get("n_persons", "--"))
    macros["\\LumLo"] = macros["\\LumHi"] = macros["\\NValScenes"] = "--"
    for k in ("Train", "Val", "Test"):
        macros[f"\\N{k}Frames"] = macros[f"\\N{k}Boxes"] = "--"
    if "lum_terciles" in subsets:
        macros["\\LumLo"], macros["\\LumHi"] = (f"{t:.1f}" for t in subsets["lum_terciles"])
    meta_p = next((p for p in (a.results / "meta.json", a.results.parent / "meta.json") if p.exists()), None)
    if meta_p:
        meta = json.loads(meta_p.read_text())
        for k in ("train", "val", "test"):
            macros[f"\\N{k.title()}Frames"] = f"{meta['counts'][k]:,}".replace(",", "{,}")
            macros[f"\\N{k.title()}Boxes"] = f"{meta['boxes'][k]:,}".replace(",", "{,}")
        macros["\\NValScenes"] = str(len(meta.get("val_scenes", [])))

    def sel(m, cond="clean", sub="all"):
        return df[(df.model == m) & (df.condition == cond) & (df.subset == sub)]

    # ---- Table: main clean results
    rows = []
    for m in models:
        c50, v50 = ms(sel(m), "AP50")
        c75, _ = ms(sel(m), "AP75")
        c, v = ms(sel(m), "AP")
        ci = boot.get("AP50", {}).get("ci", {}).get(m)
        cis = f"[{pct(ci[0])}, {pct(ci[1])}]" if ci else "--"
        rows.append(f"{NAMES[m]} & {CH[m]} & {c50} & {cis} & {c75} & {c} \\\\")
        for k, val in (("ApFifty", v50), ("Ap", v)):
            macros[macro_name(m, "clean", k)] = pct(val)
    (a.out / "tab_main.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{Detection on the LLVIP test set (clean inputs). Mean$\\pm$std over "
        f"\\NSeeds{{}} seeds; 95\\% CI from a paired image-level bootstrap.}}\\label{{tab:main}}\n"
        "\\setlength{\\tabcolsep}{3pt}\\footnotesize\n\\begin{tabular}{@{}lccccc@{}}\\toprule\n"
        "Input & Ch. & AP$_{50}$ & 95\\% CI & AP$_{75}$ & AP \\\\ \\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n\\\\[2pt]{\\scriptsize $^\\dagger$single-channel image replicated to 3 channels.}\n\\end{table}\n"
    )

    # ---- Table: illumination terciles
    rows = []
    for m in models:
        cells = []
        for sub in ("lum_low", "lum_mid", "lum_high"):
            c, v = ms(sel(m, "clean", sub), "AP50")
            cells.append(c)
            macros[macro_name(m, sub, "ApFifty")] = pct(v)
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    sz = subsets.get("sizes", {})
    (a.out / "tab_illum.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{AP$_{50}$ by scene illumination (terciles of mean visible-image luma; "
        "thresholds \\LumLo{} and \\LumHi{} on a 0--255 scale; "
        f"$n$ = {sz.get('lum_low', '--')} / {sz.get('lum_mid', '--')} / {sz.get('lum_high', '--')} images).}}\\label{{tab:illum}}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lccc@{}}\\toprule\n"
        "Input & Darkest & Middle & Brightest \\\\ \\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n"
    )

    # ---- Table: robustness to sensor degradation
    rows = []
    for m in models:
        cells = []
        for cond in ROBUST:
            c, v = ms(sel(m, cond), "AP50")
            cells.append(c.split("{")[0])  # mean only, to fit the column
            macros[macro_name(m, cond, "ApFifty")] = pct(v)
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    (a.out / "tab_robust.tex").write_text(
        "\\begin{table*}[t]\\centering\\caption{AP$_{50}$ under synthetic test-time degradation of one sensor "
        "(mean over seeds). A model is only affected by degradations of the sensors it uses.}\\label{tab:robust}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lccccccc@{}}\\toprule\n"
        "& & \\multicolumn{4}{c}{Visible camera degraded} & \\multicolumn{2}{c}{Thermal camera degraded} \\\\ "
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
        "Input & " + " & ".join(ROBUST_HDR) + " \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table*}\n"
    )

    # ---- Figure: misalignment sensitivity
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    style = {"ir": ("#b5651d", "s"), "rgbt": ("#1f6f8b", "o"), "rgbt_unpaired": ("#999999", "v"), "late_wbf": ("#6a3d9a", "^"), "rgb": ("#2ca02c", "D")}
    for m in [m for m in ("rgb", "ir", "rgbt", "late_wbf", "rgbt_unpaired") if m in models]:
        ys, es = [], []
        for k in SHIFTS:
            s = sel(m, "clean" if k == 0 else f"ir_shift{k}")
            ys.append(100 * s.AP50.mean() if len(s) else np.nan)
            es.append(100 * s.AP50.std(ddof=1) if len(s) > 1 else 0)
            macros[macro_name(m, f"shift{k}", "ApFifty")] = pct(s.AP50.mean() if len(s) else None)
        c, mk = style[m]
        ax.errorbar([2 * k for k in SHIFTS], ys, yerr=es, color=c, marker=mk, ms=3.5, lw=1.2, capsize=2, label=NAMES[m].replace(" (LWIR)", ""))
    ax.set_xlabel("Thermal offset (px at native 1280$\\times$1024)")
    ax.set_ylabel("AP$_{50}$ (%)")
    ax.set_xticks([2 * k for k in SHIFTS])
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out / "fig_misalign.pdf")
    fig.savefig(a.out / "fig_misalign.png", dpi=200)
    plt.close(fig)

    # ---- Bootstrap differences -> macros (placeholders first, so the paper compiles without results)
    for key, tag in (("AP50", "ApFifty"), ("AP", "Ap")):
        for x, y in DIFF_PAIRS:
            base = macro_name("diff", x, "vs", y, tag)
            macros[base + "Lo"] = macros[base + "Hi"] = "--"
            sx, sy = sel(x)[key], sel(y)[key]  # point estimate: difference of seed means
            macros[base + "Est"] = pct(sx.mean() - sy.mean(), signed=True) if len(sx) and len(sy) else "--"
    for key, tag in (("AP50", "ApFifty"), ("AP", "Ap")):
        for pair, d in boot.get(key, {}).get("diff", {}).items():
            x, y = pair.split("-")
            base = macro_name("diff", x, "vs", y, tag)
            macros[base + "Lo"], macros[base + "Hi"] = pct(d["ci"][0], signed=True), pct(d["ci"][1], signed=True)

    # ---- Efficiency table (measured on the laptop CPU)
    bfile = a.results / "cpu_benchmark.json"
    if bfile.exists():
        b = json.loads(bfile.read_text())
        u, f, l = (b["models"][k] for k in ("unimodal_3ch", "early_fusion_4ch", "late_fusion_2x3ch"))
        fps = lambda t: f"{1000 / t:.0f}"  # noqa: E731
        rows = [
            f"Unimodal & {u['params_M']:.3f} & {u['GFLOPs']:.2f} & {u['pytorch_fp32']['median_ms']:.1f} & {u['onnxruntime_fp32']['median_ms']:.1f} & {fps(u['onnxruntime_fp32']['median_ms'])} & {u['onnx_MB']:.1f} \\\\",
            f"Early fusion & {f['params_M']:.3f} & {f['GFLOPs']:.2f} & {f['pytorch_fp32']['median_ms']:.1f} & {f['onnxruntime_fp32']['median_ms']:.1f} & {fps(f['onnxruntime_fp32']['median_ms'])} & {f['onnx_MB']:.1f} \\\\",
            f"Late fusion$^\\ast$ &{l['params_M']:.3f} & {l['GFLOPs']:.2f} & -- & {l['onnxruntime_fp32']['median_ms']:.1f} & {fps(l['onnxruntime_fp32']['median_ms'])} & {l['onnx_MB']:.1f} \\\\",
        ]
        cpu = b["cpu"].replace("with Radeon Graphics", "").strip()
        macros["\\BenchCPU"] = cpu
        macros["\\BenchThreads"] = str(b["torch_threads"])
        macros["\\UniMs"] = f"{u['onnxruntime_fp32']['median_ms']:.1f}"
        macros["\\FusMs"] = f"{f['onnxruntime_fp32']['median_ms']:.1f}"
        macros["\\LateMs"] = f"{l['onnxruntime_fp32']['median_ms']:.1f}"
        macros["\\FusExtraParams"] = f"{(f['params_M'] - u['params_M']) * 1e6:.0f}"
        (a.out / "tab_eff.tex").write_text(
            "\\begin{table}[t]\\centering\\caption{Cost of each design on a laptop CPU (\\BenchCPU{}, batch 1, 640$\\times$512 input, "
            "FP32, median of 100 runs).}\\label{tab:eff}\n\\setlength{\\tabcolsep}{2.5pt}\\footnotesize\n"
            "\\begin{tabular}{@{}lcccccc@{}}\\toprule\n"
            "& Params & & \\multicolumn{2}{c}{Latency (ms)} & & ONNX \\\\ \\cmidrule(lr){4-5}\n"
            "Model & (M) & GFLOPs & PyTorch & ORT & FPS & (MB) \\\\ \\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\n\\\\[2pt]{\\scriptsize $^\\ast$two unimodal detectors run sequentially; box-fusion time excluded.}\n\\end{table}\n"
        )

    (a.out / "numbers.tex").write_text(
        "% auto-generated by scripts/make_tables.py -- do not edit\n"
        + "\n".join(f"\\providecommand{{{k}}}{{}}\\renewcommand{{{k}}}{{{v}}}" for k, v in sorted(macros.items()))
        + "\n"
    )
    print(f"wrote {len(macros)} macros and tables to {a.out}")


if __name__ == "__main__":
    main()
