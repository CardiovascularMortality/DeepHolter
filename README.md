# DeepHolter: Cardiovascular Mortality Prediction from 12-Lead 24-Hour Holter ECGs

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)](https://pytorch.org/)

DeepHolter is a two-stage multimodal framework for predicting **two-year cardiovascular mortality** from 12-lead, 24-hour Holter electrocardiograms (ECGs) and generating individualized explanatory reports. The study included **7,102 patients** from five medical centers, comprising a retrospective development and internal-test cohort (Center A) and an independent prospective external cohort (Centers B–E), with **59,925,744 ten-second ECG segments**.

**Stage 1** produces a patient-level cardiovascular mortality risk score from raw ECG signals, automatically extracted ECG features and patient demographics. **Stage 2** retrieves reference cases and uses a large language model (LLM) to present the Stage 1 risk classification and feature attributions in a structured report. The LLM provides **self-reported confidence** in the assigned risk classification; this is not a calibrated probability and does **not** alter the Stage 1 risk score or classification threshold.

## Repository structure

Paths below are relative to the **repository root**, which contains a second directory named `DeepHolter`:

```text
DeepHolter/                         # repository root
├── README.md
├── requirements.txt                # Conda package export (not pip requirements)
└── DeepHolter/
    ├── Stage1/
    │   ├── main.py                 # Stage 1 training/evaluation entry point
    │   ├── model.py                # Multi-bag aggregation and model wrapper
    │   ├── model_BC.py             # Signal/feature encoding and fusion
    │   ├── ECGElector.py           # Temporal bag scoring and selection
    │   ├── MyDataset_BC.py         # Input dataset and feature imputation
    │   ├── SignalEmbedding.py
    │   ├── Featurelayer.py
    │   └── TransformerEncoderlayer.py
    └── Stage2/
        └── deepseek.ipynb           # LLM report-generation notebook
```

Direct links: [Stage 1 entry point](DeepHolter/Stage1/main.py) · [Stage 1 model](DeepHolter/Stage1/model.py) · [Temporal selector](DeepHolter/Stage1/ECGElector.py) · [Stage 2 notebook](DeepHolter/Stage2/deepseek.ipynb).

## Setup

```bash
git clone https://github.com/CardiovascularMortality/DeepHolter.git
cd DeepHolter
conda create -n DeepHolter python=3.10.11 -y
conda activate DeepHolter
```

The provided [`requirements.txt`](requirements.txt) is a **Linux Conda environment export** containing Conda and pip-installed package records; it is **not** a pip-compatible requirements file. In particular, do **not** use `pip install -r requirements.txt`. A working environment requires a compatible PyTorch/CUDA installation and the Python packages imported by the scripts, including `numpy`, `pandas`, `scikit-learn`, `torchvision`, `openpyxl`, `tensorboard`, `tqdm`, `joblib` and `info-nce-pytorch`. Stage 2 additionally requires Jupyter, the `openai` Python package and access to a supported DeepSeek API. Dependency versions and GPU compatibility should be checked for the target platform.

**Important:** The released scripts contain study-specific data-path placeholders and are **not an immediately runnable end-to-end demonstration**. Configure the paths and provide appropriately formatted data before executing them. Patient-level ECG recordings, metadata and trained checkpoints are not included in this repository.

## Input data and temporal bag selection

Stage 1 uses 12-lead Holter voltage data sampled at **200 Hz**, **1,656 ECG features** per 10-second segment (12 leads × 23 measurements × 6 statistics), and patient age and sex. Recordings are partitioned into consecutive, non-overlapping 10-second segments, with acquisition-hour information retained.

For each patient, the released preprocessing constructs temporal bags as follows:

1. Group the segments by recording hour, considering up to the first 24 hourly groups. Each complete hour contains **360 ten-second segment positions**.
2. Form **360 candidate temporal bags** by collecting segments at the *same within-hour position* across the retained hours. Each candidate bag contains at most one segment per hour, or **up to 24 segments**. Shorter bags are padded and masked.
3. Divide the **360 candidate bags** into **six consecutive groups of 60 bags**, corresponding to the six non-overlapping **10-minute intervals within an hour**.
4. Score candidate bags with the learnable selection module and select **the highest-scoring candidate in each group**, masking invalid bags and segments during scoring. When each group has at least one valid candidate, this yields **six representative bags per patient**, with up to **144 valid ten-second segments** in total.

**The model selects one bag per 10-minute interval, not six bags per interval.** Each selected bag gathers observations from corresponding positions across the retained hours; it is not a single continuous 10-minute recording. The implementation is in [`ECGElector.py`](DeepHolter/Stage1/ECGElector.py), with data organization in [`main.py`](DeepHolter/Stage1/main.py). **Implementation limitation:** the released selector does not explicitly handle a group with no valid candidate; inputs should be checked for at least one valid candidate in each of the six groups.

### Stage 1: cardiovascular mortality prediction

The Stage 1 network includes a **1D ResNet-50 signal encoder**, a **12-layer Transformer feature encoder** (four attention heads per layer), sinusoidal temporal positional encoding based on acquisition hour, and a **six-layer cross-attention fusion Transformer** (eight heads per layer). Gated attention combines the six selected bag representations; a two-layer Transformer and prediction head produce the patient-level risk score.

**Entry point:** [`DeepHolter/Stage1/main.py`](DeepHolter/Stage1/main.py). After preparing the study-specific input files and updating the path placeholders in that script, the command from the repository root is:

```bash
cd DeepHolter/Stage1
python main.py
```

The script expects a patient metadata spreadsheet and separate locations for ECG segment arrays (`.npy`), per-patient ECG-feature tables (`.csv`) and recording-time tables (`*_timeinfo.xlsx`), with distinct paths for the outcome groups. In `main.py`, configure `file_paths`, `chara_paths`, `timeinfo_paths`, `cvd_file_paths`, `cvd_chara_paths`, `cvd_timeinfo_paths` and the placeholder spreadsheet path in `pd.read_excel(".xlsx")`. The code uses CUDA as written and saves model checkpoints/logs during training.

The publication reports three patient-level training/test partitions; **the released entry point contains one configured split and does not automatically reproduce all three evaluations**. The original study data, split definitions and checkpoints are needed for numerical reproduction of the published results. The repository also does not supply the upstream preprocessing and feature-extraction scripts needed to produce the expected `.npy` segment files and 1,656-feature `.csv` tables from raw Holter recordings.

### Stage 2: reference-case retrieval and LLM reporting

Stage 2 uses patient representations learned by Stage 1 to identify the **three training-set reference cases with the highest cosine similarity** to each target patient. The reporting prompt includes the target patient and reference cases, their Stage 1 risk classifications, demographics, selected ECG measurements and GradientSHAP-based feature and time-of-day attributions. The LLM is instructed to preserve the target patient's Stage 1 risk classification and explain the supplied evidence.

The LLM's reported confidence (0–100%) is a **self-reported assessment of the supplied information**, not a value computed from retrieval cosine similarity, not a Stage 1 probability, and not a calibrated measure of correctness. Reference retrieval supports the context supplied to the LLM; it does not modify the Stage 1 prediction.

**Available report-generation notebook:** [`DeepHolter/Stage2/deepseek.ipynb`](DeepHolter/Stage2/deepseek.ipynb). Open it from the repository root using Jupyter:

```bash
jupyter notebook DeepHolter/Stage2/deepseek.ipynb
```

Before running the notebook, configure its input CSV, the directory of **preassembled per-patient prompt files**, the report output directory and the DeepSeek API credentials. The notebook uses the `openai` client with the DeepSeek API endpoint and requests `model="deepseek-chat"`; the exact backend model version depends on the API service configuration. The manuscript identifies the DeepSeek-V3 model used in the study; the released API call does not pin that exact model snapshot.

**Scope of the released notebook:** it implements LLM report generation from prepared inputs; it does **not** itself construct the retrieval index, generate Stage 1 predictions or assemble the complete patient/reference-case prompt files. These upstream inputs must be prepared separately. The study evaluated three independently generated reports per patient (753 reports for 251 external-cohort patients); the notebook processes each patient once per execution and does not automate the three-repeat evaluation. **Prompt-template caveat:** the released template contains the phrase "three known high-risk patients", whereas the study retrieves the three nearest reference cases regardless of their risk classification. This template wording should be corrected in the notebook before reuse.

## Data and code availability

The source files listed above are publicly accessible in this repository. The clinical data supporting the study are not redistributed here; requests for access should be directed to the corresponding author and remain subject to institutional approval and patient-confidentiality requirements.

The released code exposes key Stage 1 components and the Stage 2 report-generation notebook, but **does not constitute a turnkey reproduction package**: study-specific data paths must be configured and the original data, split definitions, trained model parameters, raw-data preprocessing/feature-extraction pipeline and Stage 2 retrieval/prompt-preparation inputs are not provided. This distinction should be considered when comparing available source code with the manuscript's *Code availability* statement.
