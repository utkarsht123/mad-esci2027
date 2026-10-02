"""COCO-style AP for a single class, factored so that per-image matching is done once and AP can be
recomputed cheaply on any subset / bootstrap resample of images.

Matches pycocotools' protocol for one class with no crowd boxes: detections sorted by score,
greedily matched to the unmatched GT with highest IoU >= threshold, top-100 detections per image,
101-point interpolated precision, AP averaged over IoU thresholds 0.50:0.05:0.95.
"""
from __future__ import annotations

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0, 1, 101)
MAX_DETS = 100


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-12)


class ImageMatches:
    """Per-image detection scores and TP flags at every IoU threshold."""

    __slots__ = ("scores", "tp", "n_gt")

    def __init__(self, pred_boxes: np.ndarray, pred_scores: np.ndarray, gt: np.ndarray):
        order = np.argsort(-pred_scores, kind="mergesort")[:MAX_DETS]
        pb, ps = pred_boxes[order], pred_scores[order]
        iou = box_iou(pb, gt)
        tp = np.zeros((len(IOU_THRS), len(pb)), bool)
        for t, thr in enumerate(IOU_THRS):
            taken = np.zeros(len(gt), bool)
            for d in range(len(pb)):
                if len(gt) == 0:
                    break
                cand = np.where(~taken, iou[d], -1.0)
                g = int(cand.argmax())
                if cand[g] >= thr:
                    taken[g] = True
                    tp[t, d] = True
        self.scores, self.tp, self.n_gt = ps.astype(np.float32), tp, len(gt)


def ap_from_matches(ms: list[ImageMatches]) -> dict:
    n_gt = sum(m.n_gt for m in ms)
    if n_gt == 0:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan"), "R100": float("nan")}
    scores = np.concatenate([m.scores for m in ms]) if ms else np.zeros(0)
    tp = np.concatenate([m.tp for m in ms], 1) if ms else np.zeros((len(IOU_THRS), 0), bool)
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[:, order]
    ctp = np.cumsum(tp, 1)
    cfp = np.cumsum(~tp, 1)
    aps, recalls = [], []
    for t in range(len(IOU_THRS)):
        rc = ctp[t] / n_gt
        pr = ctp[t] / np.maximum(ctp[t] + cfp[t], np.finfo(float).eps)
        pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr  # monotone envelope
        idx = np.searchsorted(rc, REC_THRS, side="left")
        q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)] if len(pr) else 0, 0)
        aps.append(q.mean())
        recalls.append(rc[-1] if len(rc) else 0.0)
    return {"AP": float(np.mean(aps)), "AP50": float(aps[0]), "AP75": float(aps[5]), "R100": float(np.mean(recalls))}


def bootstrap(ms_by_model: dict[str, list[list[ImageMatches]]], n: int = 1000, seed: int = 0, keys=("AP50", "AP")):
    """Paired image-level bootstrap. ms_by_model[name] is a list over seeds of per-image matches (same image order).

    Returns {key: {name: array(n)}} of the seed-averaged metric on each resample, so differences between models are
    paired (same resampled images for every model).
    """
    rng = np.random.default_rng(seed)
    n_img = len(next(iter(ms_by_model.values()))[0])
    out = {key: {k: np.empty(n) for k in ms_by_model} for key in keys}
    for b in range(n):
        idx = rng.integers(0, n_img, n_img)
        for k, seeds in ms_by_model.items():
            rs = [ap_from_matches([s[i] for i in idx]) for s in seeds]
            for key in keys:
                out[key][k][b] = np.mean([r[key] for r in rs])
    return out
