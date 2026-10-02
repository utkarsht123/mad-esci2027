"""Evaluate every trained model on the LLVIP test split under every test condition.

Conditions are pairs (RGB transform, thermal transform). Predictions are cached per
(model, seed, transform-it-depends-on), so e.g. the RGB-only model is run once per RGB transform
and the late-fusion ensemble re-uses the unimodal RGB and thermal predictions.

Outputs (in --out):
    preds/<model>_s<seed>__<rgbtf>__<irtf>.npz   raw detections (cache; delete to recompute)
    metrics.csv                                  one row per model x seed x condition x subset
    bootstrap.json                               paired image-level bootstrap CIs for key comparisons
    coco_check.json                              pycocotools vs in-house AP on clean test (sanity check)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt.common import MODALITIES, WORK_H, WORK_W, load_test_pair, make_input, read_json, read_yolo_labels, write_json  # noqa: E402
from rgbt.metrics import ImageMatches, ap_from_matches, bootstrap  # noqa: E402

# ---------------------------------------------------------------- test-time transforms
# Severities are fixed and reported in the paper. Randomness is seeded per image, so every model sees
# exactly the same corrupted pixels.


def _rng(i, salt):
    return np.random.default_rng(1000003 * salt + i)


def rgb_tf(name: str, x: np.ndarray, i: int) -> np.ndarray:
    f = x.astype(np.float32)
    if name == "clean":
        return x
    if name == "lowlight":  # ~-2.5 EV under-exposure with gamma, plus read noise
        f = 255 * (f / 255) ** 2.2 * 0.4 + _rng(i, 1).normal(0, 6, f.shape)
    elif name == "fog":  # atmospheric scattering model, constant transmission t=0.45, airlight 0.8
        f = f * 0.45 + 0.8 * 255 * 0.55
    elif name == "noise":
        f = f + _rng(i, 2).normal(0, 25, f.shape)
    elif name == "blackout":  # sensor failure
        f = np.zeros_like(f)
    else:
        raise ValueError(name)
    return np.clip(f, 0, 255).astype(np.uint8)


def ir_tf(name: str, x: np.ndarray, i: int) -> np.ndarray:
    if name == "clean":
        return x
    if name == "blackout":
        return np.zeros_like(x)
    if name == "noise":
        return np.clip(x + _rng(i, 3).normal(0, 25, x.shape), 0, 255).astype(np.uint8)
    if name.startswith("shift"):  # horizontal mis-registration by k px at 640x512 (2k px at native 1280x1024)
        k = int(name[5:])
        out = np.zeros_like(x)
        out[:, k:] = x[:, : x.shape[1] - k]
        return out
    raise ValueError(name)


CONDITIONS = {  # name -> (rgb transform, thermal transform)
    "clean": ("clean", "clean"),
    "rgb_lowlight": ("lowlight", "clean"),
    "rgb_fog": ("fog", "clean"),
    "rgb_noise": ("noise", "clean"),
    "rgb_blackout": ("blackout", "clean"),
    "ir_noise": ("clean", "noise"),
    "ir_blackout": ("clean", "blackout"),
    "ir_shift4": ("clean", "shift4"),
    "ir_shift8": ("clean", "shift8"),
    "ir_shift16": ("clean", "shift16"),
    "ir_shift32": ("clean", "shift32"),
}
USES = {"rgb": "r", "gray": "r", "ir": "t", "rgbt": "rt", "rgbt_unpaired": "rt"}


# ---------------------------------------------------------------- prediction
def predict_all(model, modality, stems, ds, rtf, ttf, batch, device):
    from ultralytics import YOLO  # noqa: F401  (import cost only when predicting)

    boxes, scores = [], []
    for s in range(0, len(stems), batch):
        ims = []
        for j, stem in enumerate(stems[s : s + batch]):
            i = s + j
            rgb, ir = load_test_pair(ds, stem)
            ims.append(make_input(modality, rgb_tf(rtf, rgb, i), ir_tf(ttf, ir, i)))
        for r in model.predict(ims, imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False, device=device):
            b = r.boxes
            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
            scores.append(b.conf.cpu().numpy().astype(np.float32))
    return boxes, scores


def save_preds(p: Path, boxes, scores):
    p.parent.mkdir(parents=True, exist_ok=True)
    n = np.array([len(b) for b in boxes])
    np.savez_compressed(p, n=n, boxes=np.concatenate(boxes) if len(boxes) else np.zeros((0, 4)), scores=np.concatenate(scores) if len(scores) else np.zeros(0))


def load_preds(p: Path):
    z = np.load(p)
    idx = np.cumsum(np.r_[0, z["n"]])
    return [z["boxes"][a:b] for a, b in zip(idx[:-1], idx[1:])], [z["scores"][a:b] for a, b in zip(idx[:-1], idx[1:])]


def late_fusion(pr, pt, method):
    """Fuse per-image detections of the RGB-only and thermal-only detectors."""
    from ensemble_boxes import weighted_boxes_fusion

    out_b, out_s = [], []
    norm = np.array([WORK_W, WORK_H, WORK_W, WORK_H], np.float32)
    for (br, sr), (bt, st) in zip(zip(*pr), zip(*pt)):
        o_r, o_t = np.argsort(-sr, kind="mergesort")[:100], np.argsort(-st, kind="mergesort")[:100]
        br, sr, bt, st = br[o_r], sr[o_r], bt[o_t], st[o_t]
        if method == "wbf":
            if len(br) + len(bt) == 0:
                out_b.append(np.zeros((0, 4), np.float32)), out_s.append(np.zeros(0, np.float32))
                continue
            b, s, _ = weighted_boxes_fusion(
                [np.clip(br / norm, 0, 1).tolist(), np.clip(bt / norm, 0, 1).tolist()],
                [sr.tolist(), st.tolist()],
                [[0] * len(sr), [0] * len(st)],
                weights=[1, 1], iou_thr=0.55, skip_box_thr=0.001, conf_type="avg",
            )
            out_b.append((np.asarray(b, np.float32).reshape(-1, 4) * norm)), out_s.append(np.asarray(s, np.float32))
        else:  # "nms": pool both sets of boxes, class-agnostic NMS at IoU 0.7 (same as the detectors)
            import torch
            from torchvision.ops import nms

            b = np.concatenate([br, bt]).astype(np.float32)
            s = np.concatenate([sr, st]).astype(np.float32)
            keep = nms(torch.from_numpy(b), torch.from_numpy(s), 0.7).numpy() if len(b) else np.zeros(0, int)
            out_b.append(b[keep]), out_s.append(s[keep])
    return out_b, out_s


def coco_check(stems, gts, boxes, scores):
    """AP from pycocotools on the same detections, to validate rgbt.metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = [{"id": i, "width": WORK_W, "height": WORK_H} for i in range(len(stems))]
    anns, dets, k = [], [], 1
    for i, g in enumerate(gts):
        for x1, y1, x2, y2 in g:
            anns.append({"id": k, "image_id": i, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            k += 1
    for i, (b, s) in enumerate(zip(boxes, scores)):
        for (x1, y1, x2, y2), c in zip(b, s):
            dets.append({"image_id": i, "category_id": 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(c)})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    gt.createIndex()
    if not dets:
        return {"AP": 0.0, "AP50": 0.0}
    E = COCOeval(gt, gt.loadRes(dets), "bbox")
    E.evaluate(), E.accumulate(), E.summarize()
    return {"AP": float(E.stats[0]), "AP50": float(E.stats[1])}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N test images")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    from ultralytics import YOLO

    meta = read_json(a.data_root / "meta.json")
    lab = a.data_root / "rgb" / "labels" / "test"
    stems = sorted(p.stem for p in (a.data_root / "rgb" / "images" / "test").glob("*.png"))
    if a.limit:
        stems = stems[: a.limit]
    gts = [read_yolo_labels(lab / f"{s}.txt") for s in stems]
    lum = np.array([meta["test_luminance"][s] for s in stems])
    t1, t2 = np.quantile(lum, [1 / 3, 2 / 3])
    subsets = {"all": np.ones(len(stems), bool), "lum_low": lum <= t1, "lum_mid": (lum > t1) & (lum <= t2), "lum_high": lum > t2}
    print(f"{len(stems)} test images, {sum(map(len, gts))} persons; luminance terciles {t1:.1f}, {t2:.1f}")

    weights = {(m, s): a.runs / f"{m}_s{s}" / "weights" / "best.pt" for m in MODALITIES for s in a.seeds}
    weights = {k: v for k, v in weights.items() if v.exists()}
    print("models found:", sorted(f"{m}_s{s}" for m, s in weights))

    cache = a.out / "preds"

    def preds(m, s, cond):
        rtf, ttf = CONDITIONS[cond]
        rtf = rtf if "r" in USES[m] else "clean"  # a model is only affected by the sensors it uses
        ttf = ttf if "t" in USES[m] else "clean"
        p = cache / f"{m}_s{s}__{rtf}__{ttf}.npz"
        if not p.exists():
            print(f"  predict {m}_s{s} rgb={rtf} ir={ttf}", flush=True)
            model = YOLO(str(weights[(m, s)]))
            save_preds(p, *predict_all(model, m, stems, a.data_root, rtf, ttf, a.batch, a.device))
        b, sc = load_preds(p)
        return b[: len(stems)], sc[: len(stems)]

    rows, matches = [], {}  # matches[(model, seed, cond)] -> list[ImageMatches]
    systems = sorted({m for m, _ in weights}, key=list(MODALITIES).index)
    if all((("rgb", s) in weights and ("ir", s) in weights) for s in a.seeds):
        systems += ["late_wbf", "late_nms"]
    for cond in a.conditions:
        for m in systems:
            for s in a.seeds:
                if m.startswith("late"):
                    b, sc = late_fusion(preds("rgb", s, cond), preds("ir", s, cond), m.split("_")[1])
                elif (m, s) in weights:
                    b, sc = preds(m, s, cond)
                else:
                    continue
                ms = [ImageMatches(bb, ss, g) for bb, ss, g in zip(b, sc, gts)]
                matches[(m, s, cond)] = ms
                for sub, mask in subsets.items():
                    if sub != "all" and cond != "clean":
                        continue
                    r = ap_from_matches([x for x, k in zip(ms, mask) if k])
                    rows.append({"model": m, "seed": s, "condition": cond, "subset": sub, **r})
                if cond == "clean" and s == a.seeds[0]:
                    print(f"  {m:14s} clean AP50={rows[-4]['AP50']:.4f} AP={rows[-4]['AP']:.4f}")
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "metrics.csv", index=False)
    write_json(a.out / "subsets.json", {"lum_terciles": [float(t1), float(t2)], "sizes": {k: int(v.sum()) for k, v in subsets.items()}, "n_test": len(stems), "n_persons": int(sum(map(len, gts)))})

    # sanity check of the AP implementation against pycocotools (first available model, clean)
    m0, s0 = next(iter(weights))
    ours = ap_from_matches(matches[(m0, s0, "clean")])
    ref = coco_check(stems, gts, *preds(m0, s0, "clean"))
    write_json(a.out / "coco_check.json", {"model": f"{m0}_s{s0}", "ours": ours, "pycocotools": ref})
    print("AP check (ours vs pycocotools):", ours["AP"], ref["AP"], "|", ours["AP50"], ref["AP50"])

    # paired bootstrap CIs (clean, all images) for headline comparisons, seed-averaged
    boot = {}
    avail = [m for m in systems if all((m, s, "clean") in matches for s in a.seeds)]
    if a.n_boot and avail:
        dists = bootstrap({m: [matches[(m, s, "clean")] for s in a.seeds] for m in avail}, n=a.n_boot)
        for key, dist in dists.items():
            ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
            boot[key] = {"ci": {m: ci(v) for m, v in dist.items()}, "diff": {}}
            for x, y in [("rgb", "gray"), ("ir", "rgb"), ("rgbt", "ir"), ("rgbt", "rgb"), ("rgbt", "rgbt_unpaired"), ("rgbt", "late_wbf"), ("late_wbf", "ir")]:
                if x in dist and y in dist:
                    d = dist[x] - dist[y]
                    boot[key]["diff"][f"{x}-{y}"] = {"ci": ci(d), "p_le_0": float((d <= 0).mean())}
        write_json(a.out / "bootstrap.json", boot)
    print(df[(df.subset == "all") & (df.condition == "clean")].groupby("model")[["AP50", "AP"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
