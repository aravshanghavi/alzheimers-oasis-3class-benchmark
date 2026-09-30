import os
import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss

# ================================================================================
# SECTION 1: FINAL N=5 SEED RESULTS
# All experimental data is now consolidated here.
# ================================================================================

# Base directory where the loss function folders (e.g., "FA_FL", "LDAM")
# containing the raw prediction (.npz) files are located.
PREDICTIONS_BASE_DIR = "."  # Assumes the script is in the 'extra_parameters_calculation' folder

SEEDS = [42, 88, 101, 256, 2024]

# --- Final N=5 Seed Performance Scores ---
final_metrics = {
    "FA_FL": {
        "Accuracy": [78.32, 75.64, 76.06, 80.86, 73.04],
        "Macro_F1": [0.54, 0.53, 0.55, 0.64, 0.50],
        "Demented_F1": [0.34, 0.36, 0.23, 0.50, 0.31],
    },
    "LDAM": {
        "Accuracy": [78.56, 77.79, 77.29, 79.20, 74.93],
        "Macro_F1": [0.54, 0.50, 0.53, 0.50, 0.48],
        "Demented_F1": [0.32, 0.36, 0.21, 0.27, 0.36],
    },
    "FocalLoss": {
        "Accuracy": [75.18, 73.25, 74.25, 79.46, 71.83],
        "Macro_F1": [0.56, 0.51, 0.54, 0.64, 0.51],
        "Demented_F1": [0.35, 0.35, 0.24, 0.52, 0.31],
    },
    "WCE": {
        "Accuracy": [75.65, 74.03, 75.60, 79.70, 72.66],
        "Macro_F1": [0.55, 0.50, 0.56, 0.64, 0.50],
        "Demented_F1": [0.35, 0.32, 0.25, 0.51, 0.28],
    }
}


# ================================================================================
# SECTION 2: CALCULATION FUNCTIONS
# ================================================================================

def calculate_ece(probabilities, labels, n_bins=15):
    """Calculates the Expected Calibration Error (ECE) of a model."""
    n_samples = len(labels)
    if n_samples == 0: return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == labels).astype(int)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correctness[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def analyze_calibration(loss_type, seeds):
    """Loads prediction files and calculates calibration metrics for a given model."""
    brier_scores = []
    eces = []

    print(f"\n--- Analyzing Calibration for: {loss_type} ---")

    for seed in seeds:
        file_path = os.path.join(PREDICTIONS_BASE_DIR, loss_type, str(seed), "raw_predictions_for_calibration.npz")
        if not os.path.exists(file_path):
            print(f"  [Warning] Prediction file not found for seed {seed}: {file_path}")
            continue

        data = np.load(file_path)
        probabilities = data['probabilities']
        true_labels = data['labels']

        # Brier Score
        true_labels_one_hot = np.eye(probabilities.shape[1])[true_labels]
        brier = brier_score_loss(true_labels_one_hot.ravel(), probabilities.ravel())
        brier_scores.append(brier)

        # ECE
        ece = calculate_ece(probabilities, true_labels)
        eces.append(ece)

        print(f"  Seed {seed}: Brier Score = {brier:.4f}, ECE = {ece:.4f}")

    return brier_scores, eces


def analyze_paired_tests(metrics_dict, model1_name, model2_name, metric_type):
    """Performs a Wilcoxon signed-rank test between two models for a given metric."""
    model1_scores = metrics_dict[model1_name][metric_type]
    model2_scores = metrics_dict[model2_name][metric_type]

    if len(model1_scores) != 5 or len(model2_scores) != 5:
        print(f"  [Error] Cannot perform paired test. Ensure 5 seed results are entered for both models.")
        return

    stat, p_value = stats.wilcoxon(model1_scores, model2_scores)

    print(f"\n--- Paired Test: {model1_name} vs. {model2_name} on {metric_type} ---")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: The difference is statistically significant (p < 0.05).")
    else:
        print("  Result: The difference is NOT statistically significant (p >= 0.05).")


def analyze_confidence_intervals(metrics_dict, model_name, metric_type):
    """Calculates the mean, std dev, and 95% confidence interval for a metric."""
    scores = metrics_dict[model_name][metric_type]

    if len(scores) < 2:
        return

    mean = np.mean(scores)
    std = np.std(scores)
    t_critical = stats.t.ppf(0.975, df=len(scores) - 1)
    margin_of_error = t_critical * (std / np.sqrt(len(scores)))
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    # Adjust formatting for F1 scores vs. Accuracy
    if "Accuracy" in metric_type:
        print(f"  Mean ± Std Dev: {mean:.2f}% ± {std:.2f}%")
        print(f"  95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    else:
        print(f"  Mean ± Std Dev: {mean:.2f} ± {std:.2f}")
        print(f"  95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")


# ================================================================================
# SECTION 3: MAIN EXECUTION
# ================================================================================
if __name__ == '__main__':
    models_to_analyze = ["FA_FL", "LDAM", "FocalLoss", "WCE"]

    # --- Calibration Analysis ---
    print(f"\n{'=' * 80}\n### CALIBRATION ANALYSIS ###\n{'=' * 80}")
    all_calibration_results = {}
    for model_name in models_to_analyze:
        brier_scores, eces = analyze_calibration(model_name, SEEDS)
        all_calibration_results[model_name] = {"Brier": brier_scores, "ECE": eces}
        if brier_scores:
            print(f"  > {model_name} Avg Brier: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")
            print(f"  > {model_name} Avg ECE:   {np.mean(eces):.4f} ± {np.std(eces):.4f}")

    # --- Confidence Interval Analysis ---
    print(f"\n{'=' * 80}\n### CONFIDENCE INTERVAL & STABILITY ANALYSIS ###\n{'=' * 80}")
    for model_name in models_to_analyze:
        print(f"\n--- Analysis for: {model_name} ---")
        for metric_name in ["Accuracy", "Macro_F1", "Demented_F1"]:
            analyze_confidence_intervals(final_metrics, model_name, metric_name)

    # --- Paired Statistical Tests ---
    print(f"\n{'=' * 80}\n### PAIRED STATISTICAL TESTS (vs. FA-FL) ###\n{'=' * 80}")
    baselines = ["LDAM", "FocalLoss", "WCE"]
    for baseline_name in baselines:
        for metric_name in ["Accuracy", "Macro_F1", "Demented_F1"]:
            analyze_paired_tests(final_metrics, "FA_FL", baseline_name, metric_name)