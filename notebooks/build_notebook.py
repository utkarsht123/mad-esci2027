"""Writes notebooks/colab_run_all.ipynb (kept as code so the notebook is reviewable in diffs)."""
import json
from pathlib import Path

cells = [
    ("md", """# RGB vs. Grayscale vs. Thermal vs. RGB-T fusion on LLVIP — full experiment run

Runs every experiment in the paper. **Use a GPU runtime** (Runtime → Change runtime type → T4 GPU).

* Everything that must survive a disconnect (training runs, predictions, metrics) is written to Google Drive,
  so you can simply **re-run all cells** after a disconnect: finished runs are skipped and interrupted runs resume.
* Budget: ≈1 GPU-hour per training run on a T4. Seed 0 of all five inputs (≈5–6 h) is enough for a first
  complete set of tables; seeds 1–2 add error bars and take two more sessions.
* Nothing in this notebook produces numbers by hand: `evaluate.py` writes `metrics.csv`, and
  `make_tables.py` turns it into the paper's tables."""),
    ("code", """# ---- settings -------------------------------------------------------------------------------
DRIVE_DIR = "/content/drive/MyDrive/rgbt_llvip"   # persistent: runs, predictions, results
CODE_REPO = "https://github.com/utkarsht123/Military-Asset-Detection-"  # repo holding this code
CODE_BRANCH = "esci2027"                            # branch with the experiment code
CODE_SUBDIR = "esci2027"                            # folder inside the repo ("" if at repo root)
SEEDS = [0, 1, 2]                                   # seed-major order: all inputs for seed 0 finish first
MODALITIES = ["rgb", "gray", "ir", "rgbt", "rgbt_unpaired"]
LLVIP_GDRIVE_ID = "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"  # official LLVIP.zip (github.com/bupt-ai-cz/LLVIP)
LLVIP_ZIP_ON_DRIVE = f"{DRIVE_DIR}/LLVIP.zip"        # optional: put the zip here to skip the download"""),
    ("code", """from google.colab import drive
drive.mount("/content/drive")
import os, subprocess
os.makedirs(DRIVE_DIR, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# ---- code + dependencies ----------------------------------------------------------------------
%cd /content
CODE_ZIP_ON_DRIVE = f"{DRIVE_DIR}/esci2027_code.zip"  # if present, used instead of GitHub
if not os.path.exists("/content/code"):
    if os.path.exists(CODE_ZIP_ON_DRIVE):
        !unzip -q "{CODE_ZIP_ON_DRIVE}" -d /content/code
    else:
        !git clone --depth 1 -b {CODE_BRANCH} {CODE_REPO} /content/code
# zip -> code at /content/code; GitHub -> code in CODE_SUBDIR
CODE = "/content/code" if os.path.exists("/content/code/scripts") else os.path.join("/content/code", CODE_SUBDIR)
%cd {CODE}
!pip -q install -r requirements.txt
!git -C /content/code log -1 --format="code commit %h %cd" 2>/dev/null || echo "code loaded from Drive zip"
"""),
    ("code", """# ---- LLVIP download (≈3.5 GB) ------------------------------------------------------------------
# By downloading LLVIP you agree to its terms of use (academic, non-commercial):
# https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md
if not os.path.exists("/content/LLVIP"):
    if os.path.exists(LLVIP_ZIP_ON_DRIVE):
        !cp "{LLVIP_ZIP_ON_DRIVE}" /content/LLVIP.zip
    else:
        # If gdown reports a quota error: open the Drive link in a browser, "Add shortcut to Drive" / download it,
        # place LLVIP.zip at LLVIP_ZIP_ON_DRIVE and re-run this cell.
        !gdown --id {LLVIP_GDRIVE_ID} -O /content/LLVIP.zip
    !unzip -q /content/LLVIP.zip -d /content/LLVIP && rm /content/LLVIP.zip
!ls /content/LLVIP/*"""),
    ("code", """# ---- build the aligned datasets (≈20 min on 2 vCPUs; local disk, rebuilt each session) -----------
!python scripts/prepare_llvip.py --llvip /content/LLVIP --out /content/llvip_yolo --workers {os.cpu_count()} --modalities {" ".join(MODALITIES)}
!cp /content/llvip_yolo/meta.json {DRIVE_DIR}/meta.json
!df -h /content | tail -1"""),
    ("code", """# ---- train (resumable) --------------------------------------------------------------------------
RUNS = f"{DRIVE_DIR}/runs"
for seed in SEEDS:
    for m in MODALITIES:
        print(f"==== {m} seed {seed} ====", flush=True)
        r = subprocess.run(["python", "scripts/train.py", "--data-root", "/content/llvip_yolo", "--modality", m,
                            "--seed", str(seed), "--project", RUNS, "--device", "0", "--workers", str(os.cpu_count())])
        if r.returncode:
            raise SystemExit(f"training failed for {m} seed {seed}")"""),
    ("code", """# ---- evaluate every finished model under every condition (≈2 h for 3 seeds) ------------------
done_seeds = [s for s in SEEDS if all(os.path.exists(f"{RUNS}/{m}_s{s}/weights/best.pt") for m in MODALITIES)]
print("seeds with all models trained:", done_seeds)
!python scripts/evaluate.py --data-root /content/llvip_yolo --runs {RUNS} --out {DRIVE_DIR}/results --seeds {" ".join(map(str, done_seeds))} --device 0 --batch 64"""),
    ("code", """# ---- tables + a bundle to send back ----------------------------------------------------------------
!python scripts/make_tables.py --results {DRIVE_DIR}/results --out {DRIVE_DIR}/results/generated
import shutil, glob
bundle = "/content/results_bundle"
shutil.rmtree(bundle, ignore_errors=True)
shutil.copytree(f"{DRIVE_DIR}/results", bundle, ignore=shutil.ignore_patterns("preds"))
for run in glob.glob(f"{RUNS}/*_s*"):
    dst = f"{bundle}/runs/{os.path.basename(run)}"
    os.makedirs(dst + "/weights", exist_ok=True)
    for f in ("results.csv", "args.yaml"):
        if os.path.exists(f"{run}/{f}"): shutil.copy(f"{run}/{f}", dst)
    if os.path.exists(f"{run}/weights/best.pt"): shutil.copy(f"{run}/weights/best.pt", dst + "/weights")
shutil.copy(f"{DRIVE_DIR}/meta.json", bundle)
shutil.make_archive(f"{DRIVE_DIR}/results_bundle", "zip", bundle)
print("Send back:", f"{DRIVE_DIR}/results_bundle.zip")"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md"
        else {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells
    ],
}
out = Path(__file__).with_name("colab_run_all.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
