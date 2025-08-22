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
author:
- |
  \
  Xuankun Yang Shanghai Jiao Tong University\
  Shanghai, China\
  `kk-dao@sjtu.edu.cn`\
bibliography:
- content/blogs/Model_Merging_Overview/overview.bib
title: Model Merging Overview
---


# Introduction

In the rapidly evolving landscape of deep learning, pre-trained models
have emerged as a central driving force in the field of machine
learning, demonstrating remarkable performance across a diverse range of
complex tasks. However, with the continuous growth in model scale and
the increasing specialization of application scenarios, the number of
fine-tuned models for specific tasks has also grown exponentially. This
trend presents a series of practical challenges. For instance, deploying
separate models for each particular task leads to high computational
costs, enormous storage requirements, and inefficiency in multi-task
processing. Traditional solutions, such as multi-task
learning[@MultitaskLearning], typically necessitate simultaneous access
to raw training data for all tasks, which is often infeasible in
real-world scenarios due to data privacy concerns or difficulties in
data integration.

To address these challenges, model merging techniques have emerged and
rapidly become a promising research direction in the field of machine
learning. The core philosophy of model merging lies in its ability to
effectively integrate the parameters or predictions of multiple
independently fine-tuned models into a single, unified model, without
the need for time-consuming additional training processes or access to
raw training data. This approach not only significantly enhances the
computational efficiency of multi-task learning but also demonstrates
substantial potential in areas such as continual learning and few-shot
learning, with its advantages becoming particularly pronounced when
dealing with large language models (LLMs) and foundation models.
Compared to traditional multi-task learning, model merging does not
require simultaneous access to all task training data, and compared to
model ensembling [@EnsemblelearningAsurvey], the merged model is cheaper
to run at inference time.

However, model merging is not without its inherent complexities. In
high-dimensional parameter spaces, potential conflicts and interference
among different models often lead to a decline in the performance of the
merged model. Consequently, effectively balancing parameter competition
among tasks and mitigating such interference has become a focal point of
research in this domain.

This overview aims to delve into the various model merging methods that
have emerged recently in the context of large models, covering
fine-tuning strategies such as fully fine-tuning (FFT) and
parameter-efficient fine-tuning (PEFT). Specially, we primarily focuses
on model merging methods for **fully fine-tuned (FFT)** large models,
while also discussing the applicability of **parameter-efficient
fine-tuning (PEFT)** strategies. We categorize existing model merging
methods into the following four main categories: (1) **Weighted Average
Strategies**, which fuse knowledge by applying different weighting
methods to each model's parameters, including Simple
Average[@wortsmanModelSoupsAveraging2022; @ilharco2022patchingopenvocabularymodelsinterpolating; @choshen2022fusingfinetunedmodelsbetter],
Fisher Average [@matenaMergingModelsFisherWeighted2022],
RegMean[@jinDatalessKnowledgeFusion2025], Task Arithmetic
[@ilharcoEditingModelsTask2023], AdaMerging
[@yangAdaMergingAdaptiveModel2024], etc. Notably, the \"Model soups\"
method [@wortsmanModelSoupsAveraging2022] falls into this category,
demonstrating that averaging weights of multiple fine-tuned models can
improve accuracy and robustness without increasing inference time. (2)
**Sparsification Strategies**, designed to reduce task interference and
optimize model structure by sparsely configuring the weights of each
model. This includes DELLA-Merging
[@deepDELLAMergingReducingInterference2024]
,TIES-Merging[@yadav2023tiesmergingresolvinginterferencemerging],
DARE[@yuLanguageModelsAre2024], Localize-and-Stitch
[@heLocalizeandStitchEfficientModel2025], FREE Merging
[@zhengFREEMergingFourierTransform2025], TALL masks and Consensus
method[@wangLocalizingTaskInformation2024]. (3) **Subspace-based
Methods**, which approach model merging from a deeper mathematical
perspective by performing subspace decomposition on model weights to
resolve potential issues and achieve better performance. Examples
include TSV-Merging[@gargiuloTaskSingularVectors2025], Isotropic
Merging[@marczakNoTaskLeft2025], Knots(especially with
PEFT)[@stoicaModelMergingSVD2024], AdaRank
[@leeAdaRankAdaptiveRank2025], which reduces the performance gap between
merged models and fine-tuned models to nearly 1%; and
STAR[@leeSTARSpectralTruncation2025]. Furthermore, PCB Merging
[@duParameterCompetitionBalancing2024] outperforms the strongest
baseline for T5-base and T5-Large models by 1.9% and 2.1% respectively.
Revisiting Weight Averaging for Model Merging(CART) also shows that the
performance gap with traditional multi-task learning can be narrowed to
within 1-3% [@choiRevisitingWeightAveraging2025]. (4) **MoE-based
Methods**, a novel class of model merging strategies inspired by the
Mixture-of-Experts (MoE) architecture in large models, offering new
insights for model merging. This category includes SMILE
[@tangSMILEZeroShotSparse2024], Twin-Merging
[@luTwinMergingDynamicIntegration2024], which demonstrates an average
improvement of 28.34% in absolute normalized score for discriminative
tasks, and WEMoE [@tangMergingMultiTaskModels2024].

By systematically organizing and analyzing these diverse methodologies,
we aim to gain a deeper understanding of the intrinsic connections and
distinctions among different model merging approaches, uncover their
underlying theoretical foundations, and explore how they effectively
address the challenges faced by large models in multi-task applications.
We believe this overview will provide researchers in the field of model
merging with a comprehensive and insightful framework, inspiring more
in-depth future explorations in this direction.