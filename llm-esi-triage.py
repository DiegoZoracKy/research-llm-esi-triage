# ====================================================================================
# ESI CLASSIFICATION SCRIPT (V4.5)
#
# Key Features:
# - Cleaned imports for a leaner script.
# - Environment-agnostic API key handling.
# - Data leakage prevention via a comprehensive exclusion list.
# - Complete reproducibility with random seeds.
# - Visual confusion matrix plotting.
# - Detailed error tracking and analysis.
#
# Author: Diego Zoracky W. Oliveira
# Date: July 30, 2025
# ====================================================================================

# --- Imports ---
import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import tenacity

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# --- Configuration ---
@dataclass
class Config:
    """Configuration for ESI classification experiments."""
    # Data Settings
    source_data_file: str = 'MIETIC-validate-samples.csv'
    max_rows_to_process: Optional[int] = 36 # 36  # None = process all validated samples
    
    # --- CRITICAL COLUMNS ---
    ground_truth_column: str = 'acuity'
    narrative_column: str = 'tiragecase'
    
    # Model Settings
    model_name: str = "gpt-5"
    temperature: float = 1.0  # GPT-5 only supports temperature=1.0
    api_call_delay: float = 0.5
    max_retries: int = 5
    
    # --- DATA LEAKAGE PREVENTION ---
    exclude_from_prompt_cols: List[str] = field(default_factory=lambda: [
        'subject_id', 'stay_id', 'hadm_id', 'acuity', 'tiragecase',
        'invasive_ventilation', 'invasive_ventilation_beyond_1h',
        'non_invasive_ventilation', 'transfer2surgeryin1h',
        'transfer_to_surgery_beyond_1h', 'transfer_to_icu_in_1h',
        'transfer_to_icu_beyond_1h', 'transfer_within_1h', 'transfer_beyond_1h',
        'expired_within_1h', 'expired_beyond_1h', 'tier1_med_usage_1h',
        'tier1_med_usage_beyond_1h', 'tier2_med_usage', 'tier3_med_usage',
        'tier4_med_usage', 'psychotropic_med_within_120min',
        'transfusion_within_1h', 'transfusion_beyond_1h',
        'red_cell_order_more_than_1', 'intraosseous_line_placed',
        'critical_procedure', 'lab_event_count', 'microbio_event_count',
        'exam_count', 'intravenous_fluids', 'intravenous', 'intramuscular',
        'nebulized_medications', 'oral_medications', 'consults_count',
        'procedure_count', 'resources_used', 'Expert 1 Opinion',
        'Expert 2 Opinion', 'Expert 3 Opinion', 'Final Decision'
    ])
    
    # Experiment Settings
    results_base_dir: Path = field(default_factory=lambda: Path("results"))
    random_seed: int = 42
    prompt_version: str = "v4.5"
    use_tqdm: bool = True
    generate_plots: bool = True
    
    # Debugging Settings
    track_detailed_errors: bool = True
    save_raw_responses: bool = True
    
    @property
    def api_key(self) -> str:
        """Get API key from environment variables."""
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it as an environment variable."
            )
        return key
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        d = asdict(self)
        d['results_base_dir'] = str(d['results_base_dir'])
        d.pop('api_key', None)
        return d

# --- Error Types ---
class ErrorType:
    """Constants for error categorization."""
    EMPTY_NARRATIVE = "ERROR_EMPTY_NARRATIVE"
    EMPTY_RESPONSE = "ERROR_EMPTY_RESPONSE"
    UNEXPECTED_FORMAT = "ERROR_UNEXPECTED_FORMAT"
    API_FAILED = "ERROR_API_FAILED"

# --- Prompt Templates ---
SYSTEM_PROMPT = """You are an expert emergency triage nurse. Your task is to determine the Emergency Severity Index (ESI) level (1-5) for a patient.

You will be provided with:
1. Structured clinical data (which may sometimes be incomplete)
2. A triage narrative summary containing crucial clinical context

Evaluation process:
1. Review the structured data for initial assessment
2. Carefully analyze the triage narrative, giving it significant weight as it provides richer clinical context
3. Provide step-by-step reasoning for your ESI determination
4. State your final ESI level as a single number (1-5) on a new line

ESI Level Definitions:
- Level 1: Immediate life-saving intervention needed (cardiac arrest, severe respiratory distress)
- Level 2: High-risk, time-critical condition (chest pain, altered mental status, severe pain)
- Level 3: Stable, requires 2+ resources (labs + imaging)
- Level 4: Stable, requires 1 resource (simple laceration)
- Level 5: Stable, requires no resources (prescription refill)

Resources include: Labs, ECG, X-rays, CT/MRI/US, IV fluids, IV/IM/Neb meds, specialty consult, procedures"""

USER_PROMPT_TEMPLATE = """Structured clinical data:
{structured_data}

Triage narrative:
\"\"\"{narrative}\"\"\"

Please provide your clinical reasoning followed by the ESI level:"""

# --- Setup Functions ---
def setup_logging(run_dir: Path) -> logging.Logger:
    """Configure comprehensive logging for the experiment."""
    logger = logging.getLogger('esi_classification')
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(run_dir / 'experiment.log')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger

def create_run_directory(cfg: Config) -> Path:
    """Create timestamped directory for experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = cfg.model_name.replace('/', '_').replace(':', '_')
    run_name = f"{timestamp}_{model_safe}_ESI_v{cfg.prompt_version}"
    run_dir = cfg.results_base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def set_random_seeds(seed: int):
    """Set all random seeds for complete reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Data Processing ---
def load_and_prepare_data(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """Load and prepare the MIETIC dataset with validation."""
    logger.info(f"Loading data from {cfg.source_data_file}")
    
    try:
        df = pd.read_csv(cfg.source_data_file, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file '{cfg.source_data_file}' not found")
    
    logger.info(f"Total samples in dataset: {len(df)}")
    
    required_cols = [cfg.ground_truth_column, cfg.narrative_column, 'Final Decision']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one or more required columns: {required_cols}")
    
    df_valid = df[df['Final Decision'].str.upper() == 'RETAIN'].copy()
    if df_valid.empty:
        raise ValueError("No validated cases (RETAIN) found in dataset")
    
    logger.info(f"Validated samples (RETAIN): {len(df_valid)}")
    
    if cfg.max_rows_to_process and cfg.max_rows_to_process < len(df_valid):
        df_valid = df_valid.head(cfg.max_rows_to_process)
        logger.info(f"Limited to first {cfg.max_rows_to_process} samples")
    
    df_valid['original_index'] = df_valid.index
    return df_valid.reset_index(drop=True)

def format_structured_data(row: pd.Series, exclude_cols: List[str]) -> str:
    """Formats all non-excluded structured data into a readable text block."""
    lines = []
    for col, value in row.items():
        if col not in exclude_cols and pd.notna(value):
            if isinstance(value, (int, np.integer)):
                value_str = str(int(value))
            elif isinstance(value, (float, np.floating)):
                value_str = f"{value:.1f}" if col == 'temperature' else f"{value:.0f}"
            else:
                value_str = str(value).strip()
            
            if value_str:
                lines.append(f"{col}: {value_str}")
    
    return "\n".join(lines) if lines else "No structured data available"

# --- LLM Interaction ---
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.getLogger('esi_classification').warning(
        f"Retrying API call (attempt {retry_state.attempt_number})..."
    )
)
def get_llm_response(client: OpenAI, model: str, system: str, user: str, temp: float) -> str:
    """Get response from LLM with retry logic."""
    logger = logging.getLogger('esi_classification')
    logger.info(f"Making API call with model: {model}")
    logger.debug(f"System prompt length: {len(system)} chars")
    logger.debug(f"User prompt length: {len(user)} chars")
    logger.debug(f"Temperature: {temp}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temp,
        )
        logger.info(f"API call successful. Response length: {len(response.choices[0].message.content or '')} chars")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Received empty response from API")
        return content.strip()
    except Exception as e:
        logger.error(f"API call failed with error: {type(e).__name__}: {str(e)}")
        logger.error(f"Model attempted: {model}")
        raise

def extract_esi_level(response: str, detailed_errors: bool = True) -> Any:
    """Extract ESI level (1-5) from LLM response with multiple strategies."""
    if not response:
        return ErrorType.EMPTY_RESPONSE if detailed_errors else None
    
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    if lines:
        if re.match(r'^[1-5]$', lines[-1]):
            return int(lines[-1])
        last_match = re.search(r'\b([1-5])$', lines[-1])
        if last_match:
            return int(last_match.group(1))
    
    pattern_match = re.search(r'(?:ESI|Level)[\s:]+([1-5])\b', response, re.IGNORECASE)
    if pattern_match:
        return int(pattern_match.group(1))
    
    all_matches = re.findall(r'\b([1-5])\b', response)
    if all_matches:
        return int(all_matches[-1])
    
    return ErrorType.UNEXPECTED_FORMAT if detailed_errors else None

# --- Main Processing ---
def process_patient(row: pd.Series, client: OpenAI, cfg: Config, logger: logging.Logger) -> Dict[str, Any]:
    """Process a single patient case with comprehensive error handling."""
    result = {
        'original_index': row.get('original_index', row.name),
        'actual_esi': int(row[cfg.ground_truth_column]),
        'predicted_esi': None, 'status': 'pending', 'error_type': None,
        'user_prompt': None, 'response': None, 'processing_time': None
    }
    
    narrative = str(row.get(cfg.narrative_column, ''))
    if not narrative.strip():
        result.update({'status': 'error', 'error_type': ErrorType.EMPTY_NARRATIVE})
        return result

    user_prompt = USER_PROMPT_TEMPLATE.format(
        structured_data=format_structured_data(row, cfg.exclude_from_prompt_cols),
        narrative=narrative
    )
    result['user_prompt'] = user_prompt
    
    start_time = time.time()
    try:
        response = get_llm_response(client, cfg.model_name, SYSTEM_PROMPT, user_prompt, cfg.temperature)
        result['processing_time'] = time.time() - start_time
        
        if cfg.save_raw_responses:
            result['response'] = response
        
        predicted_esi = extract_esi_level(response, cfg.track_detailed_errors)
        
        if isinstance(predicted_esi, int):
            result.update({'predicted_esi': predicted_esi, 'status': 'success'})
        else:
            result.update({'status': 'error', 'error_type': predicted_esi})
            
    except Exception as e:
        result['processing_time'] = time.time() - start_time
        logger.warning(f"Error processing patient {result['original_index']}: {str(e)}")
        result.update({'status': 'error', 'error_type': ErrorType.API_FAILED})
        if cfg.save_raw_responses:
            result['response'] = str(e)
    
    return result

def run_experiment(df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """Run ESI classification on entire dataset with progress tracking."""
    client = OpenAI(api_key=cfg.api_key)
    results = []
    
    logger.info(f"Starting ESI classification with {cfg.model_name}")
    
    iterator = df.iterrows()
    if cfg.use_tqdm and HAS_TQDM:
        iterator = tqdm(iterator, total=len(df), desc="Processing patients")
    
    for idx, row in iterator:
        result = process_patient(row, client, cfg, logger)
        results.append(result)
        time.sleep(cfg.api_call_delay)
    
    results_df = pd.DataFrame(results)
    success_count = (results_df['status'] == 'success').sum()
    logger.info(f"Processing complete: {success_count} successful, {len(results_df) - success_count} errors")
    
    return results_df

# --- Visualization ---
def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: Path, logger: logging.Logger):
    """Create and save confusion matrix visualization."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib/Seaborn not available - skipping confusion matrix plot.")
        return
    
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 14}, cbar_kws={"shrink": 0.8})
        plt.title('ESI Classification Confusion Matrix', fontsize=18, pad=20)
        plt.ylabel('Actual ESI Level', fontsize=14)
        plt.xlabel('Predicted ESI Level', fontsize=14)
        plt.xticks(rotation=0); plt.yticks(rotation=0)
        
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f}', 
                 transform=plt.gca().transAxes, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix plot: {e}")

# --- Metrics and Reporting ---
def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics with error analysis."""
    valid_df = results_df[results_df['status'] == 'success'].copy()
    
    metrics = {
        'n_total': len(results_df),
        'n_successful': len(valid_df),
        'n_errors': len(results_df) - len(valid_df),
        'success_rate': len(valid_df) / len(results_df) if len(results_df) > 0 else 0
    }
    
    if 'processing_time' in results_df.columns:
        proc_times = results_df['processing_time'].dropna()
        if not proc_times.empty:
            metrics['processing_time_stats'] = {
                'mean': proc_times.mean(), 'std': proc_times.std(),
                'min': proc_times.min(), 'max': proc_times.max()
            }
    
    if metrics['n_errors'] > 0:
        metrics['error_breakdown'] = results_df[results_df['status'] == 'error']['error_type'].value_counts().to_dict()
    
    if len(valid_df) < 2:
        return metrics
    
    y_true = valid_df['actual_esi'].astype(int)
    y_pred = valid_df['predicted_esi'].astype(int)
    
    metrics.update({
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa_quadratic': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
    })
    
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    metrics['confusion_matrix'] = cm.tolist()
    
    metrics['classification_report'] = classification_report(
        y_true, y_pred, labels=[1, 2, 3, 4, 5],
        target_names=[f'ESI-{i}' for i in range(1, 6)],
        output_dict=True, zero_division=0
    )
    
    return metrics

def save_experiment_results(results_df: pd.DataFrame, metrics: Dict, run_dir: Path, cfg: Config, duration: float, logger: logging.Logger):
    """Save all experiment results with multiple formats and visualizations."""
    logger.info("Saving experiment results...")
    
    results_df.to_csv(run_dir / 'predictions.csv', index=False)
    
    metadata = {
        'experiment_info': {'date': datetime.now().isoformat(), 'duration_seconds': round(duration, 2)},
        'configuration': cfg.to_dict(),
        'prompts': {'system': SYSTEM_PROMPT, 'user_template': USER_PROMPT_TEMPLATE}
    }
    with open(run_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    if cfg.generate_plots and 'confusion_matrix' in metrics:
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), 
                              [f'ESI {i}' for i in range(1, 6)], 
                              run_dir / 'confusion_matrix.png', logger)
    
    if 'confusion_matrix' in metrics:
        cm_df = pd.DataFrame(metrics['confusion_matrix'],
                             index=[f'True_ESI_{i}' for i in range(1, 6)],
                             columns=[f'Pred_ESI_{i}' for i in range(1, 6)])
        cm_df.to_csv(run_dir / 'confusion_matrix.csv')
    
    generate_summary_report(metrics, run_dir, cfg, duration, logger)
    
    if metrics['n_errors'] > 0:
        results_df[results_df['status'] == 'error'].to_csv(run_dir / 'error_cases.csv', index=False)
    
    logger.info(f"All results saved to: {run_dir}")

def generate_summary_report(metrics: Dict, run_dir: Path, cfg: Config, duration: float, logger: logging.Logger):
    """Generate comprehensive human-readable summary report."""
    report_lines = [
        "="*70, "ESI CLASSIFICATION EXPERIMENT SUMMARY", "="*70,
        f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {cfg.model_name}", f"Duration: {duration:.1f} seconds\n",
        "--- DATASET & PROCESSING ---",
        f"Source File: {cfg.source_data_file}",
        f"Total Samples Processed: {metrics['n_total']}",
        f"Successful Predictions: {metrics['n_successful']} ({metrics['success_rate']:.1%})\n"
    ]
    
    if metrics.get('error_breakdown'):
        report_lines.append("--- ERROR BREAKDOWN ---")
        for error_type, count in metrics['error_breakdown'].items():
            report_lines.append(f"{error_type}: {count}")
        report_lines.append("")

    if 'accuracy' in metrics:
        report_lines.extend([
            "--- PERFORMANCE METRICS ---",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Quadratic Weighted Kappa: {metrics['kappa_quadratic']:.4f}\n",
            "--- CLASSIFICATION REPORT ---",
        ])
        report_dict = metrics['classification_report']
        header = f"{'':<12}{'precision':<12}{'recall':<12}{'f1-score':<12}{'support'}"
        report_lines.append(header)
        for label, values in report_dict.items():
            if isinstance(values, dict):
                p, r, f1, s = values['precision'], values['recall'], values['f1-score'], values['support']
                report_lines.append(f"{label:<12}{p:<12.2f}{r:<12.2f}{f1:<12.2f}{s}")
        
        report_lines.append("\n" + "--- CONFUSION MATRIX ---")
        cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                             index=[f"True ESI-{i}" for i in range(1, 6)], 
                             columns=[f"Pred ESI-{i}" for i in range(1, 6)])
        report_lines.append(cm_df.to_string())

    with open(run_dir / 'summary_report.txt', 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info("\n" + "\n".join(report_lines))

# --- Main Entry Point ---
def main():
    """Run the ESI classification experiment."""
    try:
        cfg = Config()
        set_random_seeds(cfg.random_seed)
        run_dir = create_run_directory(cfg)
        logger = setup_logging(run_dir)
        
        logger.info("="*70 + f"\nStarting Experiment: {run_dir.name}\n" + "="*70)
        
        df = load_and_prepare_data(cfg, logger)
        
        start_time = time.time()
        results_df = run_experiment(df, cfg, logger)
        duration = time.time() - start_time
        
        metrics = calculate_metrics(results_df)
        
        save_experiment_results(results_df, metrics, run_dir, cfg, duration, logger)
        
        logger.info(f"Experiment completed successfully in {duration:.1f} seconds.")
        
    except Exception as e:
        logging.getLogger('esi_classification').error(f"Experiment failed with unhandled error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
