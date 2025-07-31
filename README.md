# LLM Performance in ESI Triage Classification

This repository contains the code and resources for a study investigating the effectiveness of a low-cost prompt engineering strategy for Emergency Severity Index (ESI) classification using Large Language Models (LLMs).

The goal of this project is to provide a transparent and reproducible framework for evaluating the performance of LLMs in clinical triage tasks, with a rigorous focus on preventing data leakage to simulate a realistic decision-making scenario.

## 📜 About the Project

Overcrowding in emergency departments is a global challenge. This study explores how prompt engineering, a low-cost alternative to fine-tuning, can be used to guide LLMs to achieve high performance in patient classification, aligning with the clinical reasoning of experts.

The main script (`esi_classification.py`) runs an experiment on a validated subset of the MIETIC dataset, generates ESI predictions, and produces a complete set of performance reports and artifacts for analysis.

## 📄 Paper Draft

A draft of the accompanying paper for this research, "High-Performance Emergency Triage Classification Using Cost-Effective Prompt Engineering," is available for viewing. As a work-in-progress, feedback is welcome.

**[Read the Paper Draft on Google Docs](https://docs.google.com/document/d/1nWDxZZoFTt7uOHlm-gbFdwWYvWPSb-Ls0K5Jog7Oclg/edit?tab=t.0)**

### ✨ Key Features

* **Methodological Rigor:** Implements a comprehensive exclusion list to prevent data leakage of clinical outcomes, ensuring a fair evaluation.
* **Full Reproducibility:** Uses random seeds and saves all configurations, metrics, and prompts in metadata files for each run.
* **Comprehensive Reporting:** Generates multiple artifacts for each experiment, including detailed logs, raw predictions, metrics in JSON format, a human-readable summary report, and a confusion matrix visualization.
* **Robust Code:** Structured with software best practices, including dataclass configuration, professional logging, and error handling.

## 📊 Dataset

This experiment uses the **MIMIC-IV-Ext Triage Instruction Corpus (MIETIC)**, publicly available on the PhysioNet platform.

* **Source:** <https://physionet.org/content/mietic/1.0.0/>
* **File Used:** `MIETIC-validate-samples.csv`

The script automatically filters the dataset to use only the 36 cases where the `Final Decision` was validated as 'RETAIN' by experts, ensuring a high-quality ground truth.

## 🚀 Getting Started

Follow the steps below to replicate the experiment.

### ✅ Prerequisites

* Python 3.8 or higher
* An OpenAI account with API access

### ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DiegoZoracKy/research-llm-esi-triage.git
    cd research-llm-esi-triage
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    The script requires the `OPENAI_API_KEY` to be set as an environment variable.
    ```bash
    export OPENAI_API_KEY="your_key_here"
    ```

### ▶️ Running the Experiment

With everything set up, run the script from your terminal:

```bash
python esi_classification.py
```

The script will create a new directory inside the `results/` folder for each run, containing all the generated artifacts.

## 📂 Understanding the Results

For each run, a new folder will be created, for example: `results/20250731_010444_gpt-4.1_ESI_v4.5/`. Inside it, you will find:

* `predictions.csv`: The most detailed file, with a row for each patient, including the exact prompt sent, the raw LLM response, the extracted prediction, and the actual value.
* `metrics.json`: All performance metrics (accuracy, Kappa, F-1 score, etc.) in a structured JSON format.
* `metadata.json`: The complete "recipe" for the experiment, including the configuration used and the prompt templates.
* `summary_report.txt`: A human-readable summary of the results, ideal for a quick analysis.
* `confusion_matrix.png`: A visualization of the confusion matrix, ready to be used in presentations or the paper.
* `experiment.log`: A detailed log of the entire script execution, useful for debugging.
* `error_cases.csv`: If any errors occur, this file will contain the cases that failed, to facilitate analysis.

## CITATION

If you use this code or its results in your research, we kindly ask that you cite this repository directly. As the accompanying paper is not yet published, citing the software ensures that the work is properly credited.

You can use the following format:

```
Diego Zoracky W. Oliveira, et al. (2025). LLM Performance in ESI Triage Classification (Version 4.5) [Software]. Available from [https://github.com/DiegoZoracKy/research-llm-esi-triage](https://github.com/DiegoZoracKy/research-llm-esi-triage).
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
