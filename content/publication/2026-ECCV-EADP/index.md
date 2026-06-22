---
title: 'Combating Textual Noise and Redundancy: Entropy-Aware Dense Visual Token Pruning'

# Authors
# Replace placeholder names with the final camera-ready author list.
# Use `admin` for Xuankun Yang to link this publication to the local author profile.
authors:
  - Author A
  - admin
  - Author C
  - Author D
  - Author E

# Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2026-06-18T00:00:00Z'

# DOI will be added after the official version is available.
# doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2026-06-18T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list.
publication_types: ["paper-conference"]

# Publication name and optional abbreviated publication name.
publication: In *European Conference on Computer Vision (ECCV), 2026*
publication_short: In *ECCV 2026*

abstract: Visual token pruning is a crucial strategy for accelerating Vision-Language Models by compressing redundant image patches, yet existing methods often fail to preserve critical cues under dense instructions and fine-grained queries. In this work, we investigate this failure and identify two underlying bottlenecks, the widespread dispersion of textual noise that corrupts dense cross-modal scoring and the feature fragmentation inherent to standard token selection. To address these issues, we propose Entropy-Aware Dense Pruning (EADP), a framework that reformulates pruning as a structured compression problem. EADP first leverages statistical entropy to quantify and filter out textual noise, yielding a robust, fine-grained instruction relevance score. Subsequently, instead of naive Top-K selection, EADP casts token selection as a submodular maximization problem with a spatial prior, explicitly encouraging a holistic and non-redundant visual representation. Extensive experiments show that EADP improves the accuracy-efficiency trade-off of VLMs, preserving fine-grained visual cues under strict token budgets on challenging multimodal benchmarks.

# Summary. An optional shortened abstract.
summary: Entropy-aware dense visual token pruning for efficient vision-language models.

tags:
  - Visual Token Pruning
  - Vision-Language Models
  - Efficient Inference
  - MLLM
  - ECCV 2026

# Display this page in the Featured widget?
featured: true

# Custom links.
# Add paper/code links when public URLs are ready.
links:
  - name: Project Page
    url: /project/EADP/
# - type: pdf
#   url: ''
# - type: code
#   url: ''

# Featured image
# To use, add an image named `featured.jpg/png` to this page's folder.
image:
  caption: 'EADP overview'
  focal_point: 'Center'
  preview_only: false

# Associated Projects.
# Keep this empty for now to avoid duplicate project links; the Project Page
# link above already points to the EADP page.
projects: []

# Slides (optional).
slides: ""
---

### Motivation
{{< figure src="./motivation.png" alt="Motivation examples for EADP" class="eadp-figure" >}}

(a) illustrates a limitation of global guidance: it tends to attend to background regions.
(b) highlights the dispersion phenomenon caused by textual noise.
(c) reveals the issues of feature fragmentation and selection redundancy.

### Method Overview
{{< figure src="./featured.png" alt="Overview of the EADP method" class="eadp-figure" >}}

EADP acts as a lightweight, plug-and-play module that compresses the original set of visual tokens into a smaller, more informative subset before the downstream LLM consumes them.

#### Stage 1: Entropy-Aware Dense Scoring

EADP computes dense cross-modal similarities between non-EOS text tokens and visual tokens, then estimates the spatial entropy of each text token's similarity distribution. High-entropy tokens are treated as dispersed textual noise and filtered or down-weighted. The remaining low-entropy dense guidance is fused with the global EOS score to produce an instruction relevance map with both local precision and global semantic stability.

#### Stage 2: Structured Token Selection

After scoring, EADP refines the relevance map with spatial smoothing and score polarization. Gaussian smoothing propagates local structure, while polarization sharpens core visual entities against the background. Instead of selecting tokens with naive Top-K, EADP formulates token selection as a facility-location submodular maximization problem, encouraging non-redundant coverage of the original visual content.

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

**More results are provided in our paper.**