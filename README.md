# Visible, Thermal, or Both? — ESCI 2027 experiments and paper

A controlled study of input modality for lightweight person detection on the co-registered
[LLVIP](https://github.com/bupt-ai-cz/LLVIP) dataset. Five YOLOv8n detectors share one architecture,
initialization, recipe and split; only the input changes:

| Input | Channels | What it tests |
|---|---|---|
| `rgb` | 3 | visible baseline |
| `gray` | 1→3 | contribution of colour (H1) |
| `ir` | 1→3 | thermal vs visible (H2) |
| `rgbt` | 4 (R,G,B,T) | paired early fusion (H3) |
| `rgbt_unpaired` | 4 | same model, thermal channel from a *different* scene: does fusion need pairing? (H4) |
| `late_wbf`, `late_nms` | 3 + 3 | late fusion of the `rgb` and `ir` detectors (no extra training) |

Every model is evaluated on clean data, per illumination tercile, under synthetic degradation/failure of either
sensor (H5) and under thermal misregistration (H6), over 3 seeds with paired bootstrap CIs.

## Layout

```
rgbt/common.py            modality definitions, input construction (shared by prep/eval)
rgbt/metrics.py           COCO-style AP (matches pycocotools exactly) + paired bootstrap
scripts/prepare_llvip.py  LLVIP -> five aligned YOLO datasets, scene-level val split
scripts/train.py          one run = one modality x one seed (resumable, fixed recipe)
scripts/evaluate.py       all models x all test conditions -> results/metrics.csv, bootstrap.json
scripts/benchmark_cpu.py  params / GFLOPs / CPU latency (PyTorch + ONNX Runtime)
scripts/make_tables.py    results/ -> paper/generated/*.tex (tables, figure, number macros)
notebooks/colab_run_all.ipynb   runs everything on a Colab GPU, results persisted to Drive
paper/main.tex            IEEE conference paper; every number is a macro from generated/numbers.tex
AUDIT.md                  what was wrong in the previous draft/repo and what replaced it
```

## Reproduce

1. **GPU part (Colab, T4):** open `notebooks/colab_run_all.ipynb`, set `CODE_REPO`/`CODE_BRANCH`, *Run all*.
   ≈1 GPU-hour per training run (15 runs for 3 seeds). Re-running after a disconnect resumes where it stopped.
   The last cell writes `results_bundle.zip` to Drive.
2. **CPU benchmark (laptop):** `python -m scripts.benchmark_cpu` → `results/cpu_benchmark.json`.
3. **Paper:** unzip the bundle into `results/`, then
   ```
   python scripts/make_tables.py --results results --out paper/generated
   cd paper && tectonic -X compile main.tex        # or upload paper/ to Overleaf
   ```
   Sentences in orange (`\pending{}`) state a direction of effect and must be checked against the final
   tables before `\pendingfalse` is set for submission.

Local smoke test (no GPU, synthetic data — numbers meaningless):
```
python scripts/make_synthetic_llvip.py --out /tmp/smoke/LLVIP
python scripts/prepare_llvip.py --llvip /tmp/smoke --out /tmp/smoke/yolo
python scripts/train.py --data-root /tmp/smoke/yolo --modality rgbt --project /tmp/smoke/runs --device cpu --epochs 2 --imgsz 320 --batch 8 --workers 0
```

## Notes on fairness

* All inputs are derived from the same resized (640×512) pixels; 3-channel variants are lossless PNG, 4-channel TIFF.
* HSV colour jitter is off for **every** run (Ultralytics cannot apply it to 4-channel input).
* 4-channel init: pretrained RGB filters + their mean on the thermal channel, scaled by 3/4.
* Validation = whole held-out scenes from LLVIP train (no near-duplicate leakage); test = official LLVIP test.
* The unpaired model is always *tested* on correctly paired frames.
