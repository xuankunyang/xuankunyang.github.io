---
title: EADP
date: 2026-06-18
url: /project/EADP/
summary: Entropy-aware dense visual token pruning for efficient vision-language models.
authors:
  - admin
tags:
  - Visual Token Pruning
  - Vision-Language Models
  - Efficient Inference
  - MLLM
  - ECCV 2026
image:
  caption: 'EADP overview'
  focal_point: 'Center'
  preview_only: false
---

## Combating Textual Noise and Redundancy: Entropy-Aware Dense Visual Token Pruning

**ECCV 2026**

**Authors:** Xuehui Wang<sup>1,*</sup>, Xuankun Yang<sup>1,*</sup>, Wei Shen<sup>1,†</sup>

**Affiliations:** <sup>1</sup>[Shanghai Jiao Tong University](https://www.sjtu.edu.cn/), China

<sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.

[Code](https://github.com/SJTU-DeepVisionLab/EADP) | [BibTeX](#citation)

> **TL;DR:** EADP is a plug-and-play visual token pruning framework for VLMs/MLLMs. It combines entropy-aware dense scoring with submodular token selection to preserve fine-grained visual cues under strict token budgets.

<!-- TODO: Replace author list, affiliations, venue metadata, links, and figures after the camera-ready/project assets are finalized. -->

### Abstract

Visual token pruning is a crucial strategy for accelerating Vision-Language Models by compressing redundant image patches, yet existing methods often fail to preserve critical cues under dense instructions and fine-grained queries. In this paper, we investigate this failure and identify two underlying bottlenecks: the widespread dispersion of textual noise that corrupts dense cross-modal scoring, and the feature fragmentation inherent to standard token selection. To address these issues, we propose **E**ntropy-**A**ware **D**ense **P**runing (EADP), a framework that reformulates pruning as a structured compression problem. EADP first leverages statistical entropy to quantify and filter out textual noise, yielding a robust, fine-grained instruction relevance score. Subsequently, instead of naive Top-$K$ selection, EADP casts token selection as a submodular maximization problem with a spatial prior, explicitly guaranteeing a holistic and non-redundant visual representation. Extensive experiments demonstrate that EADP significantly improves the accuracy-efficiency trade-off of VLMs, robustly preserving fine-grained visual cues under strict token budgets while achieving state-of-the-art performance on challenging multimodal benchmarks.

### Motivation
{{< figure src="./motivation.png" alt="Motivation examples for EADP" class="eadp-figure" >}}

(a) illustrates a limitation of global guidance: it tends to attend to background regions.
(b) highlights the dispersion phenomenon caused by textual noise.
(c) reveals the issues of feature fragmentation and selection redundancy.

<!-- TODO: Replace this table with the paper's motivation figure or a web-optimized recreation. -->

### Method Overview
{{< figure src="./featured.png" alt="Overview of the EADP method" class="eadp-figure" >}}

EADP acts as a lightweight, plug-and-play module that compresses the original set of visual tokens into a smaller, more informative subset before the downstream LLM consumes them.

#### Stage 1: Entropy-Aware Dense Scoring

EADP computes dense cross-modal similarities between non-EOS text tokens and visual tokens, then estimates the spatial entropy of each text token's similarity distribution. High-entropy tokens are treated as dispersed textual noise and filtered or down-weighted. The remaining low-entropy dense guidance is fused with the global EOS score to produce an instruction relevance map with both local precision and global semantic stability.

#### Stage 2: Structured Token Selection

After scoring, EADP refines the relevance map with spatial smoothing and score polarization. Gaussian smoothing propagates local structure, while polarization sharpens core visual entities against the background. Instead of selecting tokens with naive Top-K, EADP formulates token selection as a facility-location submodular maximization problem, encouraging non-redundant coverage of the original visual content.

<!-- TODO: Add the final method diagram and, if useful, a short pseudocode block adapted for the project page. -->

### Results

#### Results on LLaVA-1.5
{{< figure src="./llava_1p5_7b.png" alt="Results on LLaVA-1.5" class="eadp-figure" >}}

#### Results on LLaVA-1.6
{{< figure src="./llava_1p6_7b.png" alt="Results on LLaVA-1.6" class="eadp-figure" >}}

#### Results on Qwen2.5-VL
{{< figure src="./qwen2p5_vl_7b.png" alt="Results on Qwen2.5-VL" class="eadp-figure" >}}

#### Results on LLaVA-Video
{{< figure src="./llava_video_7b.png" alt="Results on LLaVA-Video" class="eadp-figure" width="75%" >}}

#### Efficiency Analysis
{{< figure src="./efficiency_analysis.png" alt="Efficiency analysis" class="eadp-figure" width="75%" >}}

<!-- TODO: Replace the placeholder table with the final public results and add visual comparisons. -->

**More results are provided in our paper.**

### Citation

```bibtex
@inproceedings{wang2026eadp,
    title     = {Combating Textual Noise and Redundancy: Entropy-Aware Dense Visual Token Pruning},
    author    = {Wang, Xuehui and Yang, Xuankun and Shen, Wei},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2026},
    url       = {https://github.com/SJTU-DeepVisionLab/EADP}
}
```
