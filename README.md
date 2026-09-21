# EEG-Based Depression Severity Classification Using Resting-State EEG

A research-oriented machine-learning project for classifying **three levels of depression severity** from participant-level resting-state, eyes-closed EEG-derived features in adolescents.

> **Research disclaimer:** This project is an experimental research prototype for educational, research, and AI-competition purposes. It is not a medical device, diagnostic system, or substitute for assessment by a qualified mental-health professional.

## Project Objective

The objective is to investigate whether objective EEG-derived features can support the classification of three depression-severity categories:

- **Minimal depression** — Group 1
- **Mild depression** — Group 2
- **Moderate depression** — Group 3

The current modeling pipeline uses the **eyes-closed EEG condition**. Participant questionnaire scores, including BDI-II and PHQ-9, are used only for metadata inspection and label validation. They are deliberately excluded from the predictive feature matrix so that questionnaire information does not leak into the EEG classifier.

## Dataset Summary

| Property | Description |
|---|---|
| Participants | 85 adolescents |
| EEG condition modeled | Resting-state, eyes closed |
| Input representation | 600 EEG-derived features per participant |
| Prediction unit | One row per participant |
| Target | Three-class depression severity |
| Minimal group | 30 participants |
| Mild group | 27 participants |
| Moderate group | 28 participants |
| Held-out test set | 17 participants |

The participant-level design is essential. Multiple rows or segments from the same participant must not be distributed between training and testing because this can create participant-level leakage and produce overly optimistic performance estimates.

## Main Result

The selected model was an **Extra Trees classifier** within a leakage-controlled scikit-learn pipeline. On one stratified held-out test split containing 17 participants, it achieved:

| Metric | Held-out test result |
|---|---:|
| Accuracy | **64.7%** |
| Balanced accuracy | **64.4%** |
| Macro precision | **65.8%** |
| Macro recall | **64.4%** |
| Macro F1-score | **63.8%** |
| Correct predictions | **11 / 17** |

These values describe one held-out participant split. They should not be interpreted as definitive evidence of clinical performance.

The class-wise F1-scores were:

| Class | F1-score |
|---|---:|
| Minimal | 0.55 |
| Mild | 0.80 |
| Moderate | 0.62 |

## Stability Analysis

Repeated stratified cross-validation was used to examine how performance changes across participant splits. This analysis is reported separately from the final held-out test result.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Balanced accuracy | 52.1% | 11.3 percentage points |
| Macro F1-score | 50.6% | 11.8 percentage points |

The difference between the held-out test result and the repeated-split stability estimate demonstrates why a single accuracy value should not be reported without uncertainty or variability information. The small sample size makes the estimates sensitive to the particular participants included in each split.

## Methodology

The project is organized into two main analytical stages.

### Exploratory Data Analysis and Preprocessing

The EDA stage loads the participant-characteristics table and EEG spreadsheet files, checks data quality, reshapes the EEG measurements into a participant-level table, constructs the three-class target, and exports the clean modeling dataset.

The EDA notebook includes:

- Participant metadata inspection
- Data-type and dimensionality checks
- Duplicate-identifier checks
- Missing-value analysis
- Participant-level EEG reshaping
- Label construction and class-count analysis
- EEG feature distributions
- EEG band-frequency summaries, where applicable
- Descriptive outlier analysis
- Descriptive correlation analysis
- Feature heatmaps
- Export of the final participant-level modeling table

The descriptive analyses are not treated as evidence of clinical causation. Questionnaire scores are not used as predictive EEG inputs.

### Modeling

The modeling workflow uses a stratified participant-level train/test split. The training portion is used for model comparison and hyperparameter tuning, while the test participants remain untouched until final evaluation.

The final pipeline is:

```text
SimpleImputer(strategy="median")
        ↓
SelectKBest feature selection
        ↓
ExtraTreesClassifier(class_weight="balanced")
Imputation and feature selection are placed inside the scikit-learn pipeline. Therefore, these operations are fitted separately within each training fold during cross-validation rather than using information from the full dataset.
Baseline comparison models include:
Logistic Regression
Support Vector Machine
Extra Trees
Extra Trees was selected for the final reported pipeline because it provided the strongest overall performance among the evaluated configurations under the leakage-controlled workflow. This selection does not imply that Extra Trees is universally optimal for other EEG datasets.
Leakage-Controlled Design
The following safeguards are used:
The modeling unit is one participant rather than an individual EEG segment.
Participants are split into training and test sets before final evaluation.
The held-out test set is not used for hyperparameter tuning or model selection.
Imputation is fitted inside the modeling pipeline.
Feature selection is fitted inside the modeling pipeline.
Hyperparameter tuning uses training data and cross-validation only.
BDI-II and PHQ-9 scores are excluded from the EEG feature matrix.
Held-out test metrics and repeated-split stability estimates are reported separately.
Repository Structure
The recommended project organization is:
text
EEG_Depression_Project/
│
├── README.md
│
├── EDA/
│   ├── 01_EDA_Preprocessing_Professional.ipynb
│   ├── Eyes-closed EEG.xlsx
│   ├── Eyes-open EEG.xlsx
│   ├── Participant characteristics.xlsx
│   ├── final_df.csv
│   └── eda_figures/
│
├── Modeling/
│   ├── Modeling.ipynb
│   ├── final_df.csv
│   ├── final_participant_predictions.csv
│   └── final_model_summary.csv
│
├── Streamlit_Demo/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── final_df.csv
│
├── Reports/
│   ├── Report_of_AICompetition_Final.docx
│   └── AI_Driven_Depression_Diagnostics_Fact_Checked.pptx
│
└── figures/
    └── exported publication-ready figures
The eyes-open file may be retained for data documentation or future analysis, but the current reported model uses the eyes-closed condition only.
Reproducing the Analysis
1. Run EDA and preprocessing
Open:
text
EDA/01_EDA_Preprocessing_Professional.ipynb
Run the notebook from top to bottom. Update the input paths in the configuration section if the spreadsheet files are stored elsewhere. The notebook exports the cleaned participant-level dataset, typically as:
text
EDA/final_df.csv
2. Run modeling
Open:
text
Modeling/Modeling.ipynb
Make sure that the notebook can access final_df.csv. Run the cells in order. The notebook produces model metrics, classification reports, confusion matrices, participant-level predictions, and summary tables.
Expected output files include:
text
Modeling/final_participant_predictions.csv
Modeling/final_model_summary.csv
Running the Streamlit Demo
The Streamlit application is an interactive research demonstration. It displays a model prediction and class probabilities for participant-level EEG features. It must not be used for diagnosis or treatment decisions.
Windows PowerShell
From the project root, use the Python interpreter associated with the environment where Streamlit is installed:
powershell
cd EEG_Depression_Project
& "C:\Users\<YourUser>\miniconda3\envs\ai311\python.exe" -m pip install -r .\Streamlit_Demo\requirements.txt
& "C:\Users\<YourUser>\miniconda3\envs\ai311\python.exe" -m streamlit run .\Streamlit_Demo\app.py
If pip, py, or conda is not recognized as a command, use the full path to the Python executable as shown above.
Standard Python installation
Bash
cd EEG_Depression_Project
python -m pip install -r Streamlit_Demo/requirements.txt
python -m streamlit run Streamlit_Demo/app.py
After the application starts, open the local address shown in the terminal. It is usually:
text
http://localhost:8501
Keep the terminal open while using the application. Press Ctrl+C in the terminal to stop the demo.
Reproducibility Notes
For reproducible results, keep the following settings fixed unless a new experiment is intentionally being conducted:
Random seeds
Participant-level split strategy
Feature-selection settings
Cross-validation strategy
Hyperparameter search grid
Dataset version
EEG preprocessing rules
Every reported metric should be labeled according to the estimate it represents:
Held-out test result: performance on the final untouched test participants.
Cross-validation estimate: average performance across training folds.
Stability estimate: mean and variability across repeated participant splits.
These estimates answer different questions and must not be presented as interchangeable.
Limitations
The conclusions are limited by the following factors:
The sample contains only 85 participants.
The held-out test set contains only 17 participants.
Performance varies across participant splits.
The dataset represents one study population and acquisition setting.
Only the eyes-closed condition is modeled in the reported pipeline.
The input consists of processed EEG-derived features rather than raw-waveform deep learning.
No independent external validation cohort is included.
The results do not establish diagnostic validity, clinical utility, or causality.
Future work should evaluate larger and more diverse cohorts, external validation datasets, probability calibration, subgroup robustness, alternative EEG representations, and interpretable model explanations.
Dataset and Methodological References
The adolescent resting-state EEG dataset and its associated publication are cited below. The original public notebook was used as a methodological reference, but the present project adapts the workflow to a three-class adolescent dataset and applies participant-level leakage controls.
Author
Shereen Shaban

Computer Science and Artificial Intelligence Student

Helwan University
References
[1] Resting-state EEG datasets of adolescents with mild, minimal, and moderate depression

[2] OSF repository for the adolescent resting-state EEG dataset

[3] EEG-Based Depression Screening public methodological notebook

Responsible Use
Predictions from this project should be treated as outputs of a research model under a specific dataset and preprocessing pipeline. They are not medical conclusions. Any future clinical application would require prospective validation, independent replication, clinical oversight, privacy protection, fairness assessment, and appropriate regulatory review.
License
No separate software license is currently declared. Before adding a license, confirm that redistribution of the code, figures, and any accompanying data-derived files is consistent with the terms of the underlying dataset and source publication. The original dataset is not included in this repository.


