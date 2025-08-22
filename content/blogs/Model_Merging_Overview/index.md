---
abstract: |
  Model merging has emerged as a technique that merges the parameters or
  predictions of multiple pre-trained models into a single one. It is a
  promising approach for unifying independently fine-tuned models into
  an integrated framework, significantly enhancing computational
  efficiency in multi-task learning. However, model merging on
  large-scale deep learning models (e.g., LLMs and foundation models)
  faces several challenges, including high computational cost,
  high-dimensional parameter space, interference between different
  heterogeneous models, etc. In this overview, we foucus on model
  merging methods on large models, involving fine-tune strategies like
  FFT(fully fine-tune) and PEFT(parameter efficient fine-tune).
  Specifically, we We divide existing model merging methods into four
  categories: (1)\"Weighted average\". which uses different methods to
  weight each model and averages them; (2)\"Sparsification strategy\"
  sparsely configure the weights of each model for subsequent
  processing; (3)\"Subspace-based methods\" perform subspace
  decomposition on model weights and resolve potential issues to achieve
  better performance; (4)\"MoE-based methods\", a novel model merging
  strategy which imitates the MoE architecture in the large models and
  provides a new idea for model merging. Our overview is helpful in
  deeply understaning the correlation between different model merging
  methods, which can enlighten the research in the field of model
  merging.
authors:
  - admin
date: 2025-08-23
title: Model Merging Overview
tags:
  - Model Merging
  - Machine Learning
image:
  caption: 'AdaMerging~'

commentable: true
---


Please refer to [this](content/blogs/Model_Merging_Overview/overview.pdf) ariticle for details.