## Prediction of Cardiovascular Mortality from 12-Lead 24-Hour Holter Electrocardiograms by Transformer and Large Language Model

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch-red)](https://pytorch.org/)



**Abstract:** We introduce DeepHolter, an interpretable multimodal framework combining Transformer and Large Language Model for 2-year cardiovascular mortality prediction. DeepHolter was developed and validated on the largest-volume dataset with 59,925,744 12-lead 10-s Holter recordings (24-hour) from 7,102 patients, including two retrospective internal cohorts and an independent prospective external cohort from five centers. 



## About DeepHolter

_**What is DeepHolter?**_ DeepHolter is an interpretable multimodal model that integrates Transformer and Large Language Model (LLM) architectures for 2-year cardiovascular mortality prediction with explainable confidence scores, which fuses multi-time-scale electrophysiological and clinical information from 12-lead, 24-hour Holter voltage data for inpatients. The model consists of two stages: (1) Cardiovascular mortality prediction, comprising an election module, a signal encoder, a feature encoder, a fusion encoder, a circadian positional encoding (PE) block, and a prediction head to yield a 2-year cardiovascular mortality risk score; (2) Interpretation & confidence estimation, which integrates cosine-similarity-based case retrieval and SHAP-guided prompting to generate individualized explanatory reports and risk-confidence scores.



## Installation

First, clone the repo and cd into the directory:
```shell
git clone https://github.com/CardiovascularMortality/DeepHolter.git
cd DeepHolter
```
Then create a conda env and install the dependencies:
```shell
conda create -n DeepHolter python=3.10.11 -y
conda activate DeepHolter
pip install -r requirements.txt
```



## Data Access & Usage

DeepHolter fuses 24-hour Holter (sampled at 200 Hz), 1,656 extracted ECG features, and patient demographics (age, sex) as input.  For computational tractability, the 24-hour Holter voltage recordings are partitioned into consecutive 10-s ECG segments, yielding patient-specific temporal sequences for model input. The consecutive 10-s Holter ECGs are then first organized into 360 temporal bags, each corresponding to the same minute-level offset across hours.

Researchers may request access to the data by contacting the corresponding author. Alternatively, users may adapt this codebase to their own datasets by modifying parameters in `Stage1//main.py` and `Stage1//model.py`. 



## Model development

### Stage 1: Cardiovascular mortality prediction

```bash
cd Stage1 
python main.py
```

Stage 1 integrates 1D ResNet-50 and Transformer in multi-instance learning for cardiovascular mortality prediction, including:

- **Election module**: a learnable module selects 6 representative bags from each non-overlapping 10-minute interval.  

- **Signal encoder**: a 1D ResNet-50 backbone is used to encode raw Holter voltage data.
- **Feature encoder:** a self-attention Transformer encoder processes structured inputs, including patient-level demographics (age and sex) and segment-level ECG features.
- **Circadian positional encoding**: sinusoidal time-based positional embeddings are applied based on the acquisition hour of each Holter segment.
- **Fusion encoder**: a 6-layer cross-attention Transformer encoder (8 attention heads per layer) integrated the signal and feature encoders.
- **Temporal aggregation**: segment-level  \<CLS\> tokens were aggregated across the full recording via a gated attention mechanism. 
- **Prediction head:** a two-layer Transformer encoder followed by a linear layer with sigmoid activation outputs the 2-year cardiovascular mortality risk score.

### Stage 2: Interpretation & confidence estimation

```bash
run Stage2//deepseek.ipynb
```

Stage 2 incorporates LLM with in-context learning guided by similar cases and feature interpretability, yielding individualized explanatory reports and risk-confidence scores.   

Patient-level representations `out_embed` from are used to retrieve the three most similar reference cases from train set via cosine similarity.

A structured prompt is constructed for the LLM (DeepSeek-V3-032417), containing: 

- a task instruction defining the analytical objective and output format; 

- descriptions of all input variables; 

- domain knowledge from GradientSHAP18 analyses, including ranked importance scores for ECG features, hourly importance, and demographics. For each retrieved patient, the prompt included their age, sex, top 20 influential ECG features (with summary statistics), and hourly importance. 

The LLM processed this prompt along with the model’s predicted risk score to produce a structured report. The report summarizes the target and reference patients, analyzes feature contributions based on SHAP values, interprets temporal risk patterns, and provides a final numerical reasoning that explains the risk prediction and assigns a confidence score based on the similarity of retrieved cases.  
