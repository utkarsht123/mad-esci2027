# Audit of the previous draft ("Cross-Modal Attention Fusion for Military Asset Detection") and repository

Checked on 2026-09-29 against `utkarsht123/Military-Asset-Detection-` (main) and `ut_csiv (2).pdf`.

## 1. Results with no supporting experiment

| Claim in the draft | What the repository contains |
|---|---|
| Cases 1–2: YOLOv8n on KAIST, RGB 85.2 / Gray 78.5 / day 88.0 / night 5.0 / fusion 90.0 % mAP@0.5 | No KAIST data, no YOLOv8 training code, no logs or result files. |
| "Fusion RGB+T (Different Scenes) ~80 / ~65 / ~70" | Not defined or implemented anywhere. |
| Fog / smoke / camouflage results (15 %, 30 %, >85 %) | KAIST has no fog or camouflage labels; no such evaluation exists. |
| 24-hour performance curve with thermal-crossover dips | KAIST has no per-hour timestamps usable this way; no code produces it. |
| LMM-DETR: EfficientNet-B0 + thermal CNN + sensor MLP, 236-D fusion, RT-DETR with 50 queries, 79–82 % mAP | Code uses **MobileViT**, no transformer decoder, no sensor MLP, no 236-D layer. The model pools the whole image to one vector and predicts **one box per image**. No mAP is computed anywhere. |
| TinyCLIP zero-shot 63–66 % | `tinyclip_integration.py` loads ViT-B-32 CLIP (not TinyCLIP); no evaluation of accuracy. |
| ~400 ms / 43 MB / ~6 GB RAM (and "<4 GB" in §4.1) | Contradictory; the INT8 model in the repo is a 32×32 3-class `SimpleCNN` calibrated on `torch.randn` data, unrelated to the detector. |

## 2. Code problems in the repository

* `efficient_multimodal_dataset.py` pairs every AU-AIR RGB image with an **unrelated HIT-UAV thermal image**
  selected by a cycling counter (`self._thermal_idx`). The "multimodal" model therefore never saw a paired RGB-T sample.
  (The new study turns exactly this situation into a controlled experiment: `rgbt_unpaired`.)
* `essential_loss_functions.DetectionLoss` supervises only the first ground-truth box of each image.
* `ablation_studies_mini.py`, `baseline_comparison_lite.py`, `cpu_performance_tests.py`,
  `performance_analysis_lite.py` run on `sklearn.make_classification` synthetic data.
* `dashboard.py` displays hard-coded dummy accuracies (0.85, 0.92, …).
* `.gitattributes` contains unresolved merge-conflict markers.

## 3. Text-level errors in the draft

* Thermal-only "overall 83.5 %" is not the mean of 82.0 (day) and 89.0 (night).
* Case 3 is described as VEDAI in §3.2 and as AU-AIR/HIT-UAV elsewhere.
* YOLOv8's backbone is described as CSPDarknet53 (it is a C2f-based CSP variant).
* Reference list numbering has gaps (4, 6–8, 11–13, …) and YOLOv8 is cited as [14] (the KAIST paper).
* Title and framing say "military asset detection", but all claimed data are civilian pedestrians/vehicles.

## 4. References that could not be verified

Crossref (2026-09-29) finds no publication matching these entries of the draft:

* [3] Chen, Li, Wang, "Cross-modal DETR for vision-language object detection," CVPR 2022
* [19] Li, Xu, Wang, "Adaptive cross-attention for RGB-thermal detection," IEEE TIP 30, 2021
* [26] Liu et al., "Misalignment-robust multispectral pedestrian detection," IEEE T-IV 5(2), 2020
* [29] Militão et al., "A survey on multispectral pedestrian detection," IVC 83, 2019
* [32] Wagner, Zhou, Lin, "Late fusion for thermal and visual military vehicle recognition," JDMS 18(4), 2021
* [33] Wang et al., "Graph attention fusion for RGB-thermal pedestrian detection," IEEE TIP 30, 2021
* [35] Wu, Sun, Zhao, "Benchmarking multimodal DETR for RGB-thermal object detection," PR 134, 2023
* [1] Amirkolaee, Arefi, "Decision-level fusion of thermal and RGB images…," IPT 99, 2019
* [9] Guan, Huang, Lu, "Multi-modal interaction networks for RGB-thermal pedestrian detection," PR 112, 2021

Misattributed: [25] "Liu, Yuen, Torralba: KAIST multispectral pedestrian dataset, CVPR 2016" — those authors wrote
a scene-parsing paper; KAIST is Hwang et al., CVPR 2015 (already [14]). [21] DN-DETR is Li, Zhang, Liu, Guo, Ni,
Zhang, **CVPR 2022**, not "Li, Wang, Zhang, NeurIPS 35". [2] DETR (Carion et al.) is cited for quantized YOLO /
EfficientNet backbones, which it does not cover.

The new `paper/refs.bib` contains only entries verified against Crossref (DOIs included).

## 5. What replaced it

See `README.md`: a controlled five-input study on LLVIP with paired-vs-unpaired fusion, late fusion, illumination
terciles, sensor degradation/failure, misregistration, multi-seed statistics and measured CPU cost. Every number in
`paper/main.tex` is generated from result files by `scripts/make_tables.py`. The earlier claims about fog,
camouflage, 24-hour operation, LMM-DETR and TinyCLIP were dropped because no experiment supports them.
