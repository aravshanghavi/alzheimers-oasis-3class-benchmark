import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# ================================================================================
# SECTION 1: FINAL N=5 SEED RESULTS
# ================================================================================

PREDICTIONS_BASE_DIR = "."
SEEDS = [42, 88, 101, 256, 2024]

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
# SECTION 2: CALCULATION & PLOTTING FUNCTIONS
# ================================================================================

def calculate_ece(probabilities, labels, n_bins=15):
    """Calculates the Expected Calibration Error (ECE) of a model."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == labels)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correctness[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def analyze_calibration(loss_type, seeds):
    """
    Loads prediction files, calculates metrics, and returns aggregated predictions.
    """
    brier_scores, eces = [], []
    all_probas, all_labels = [], []

    print(f"\n--- Analyzing Calibration for: {loss_type} ---")

    for seed in seeds:
        file_path = os.path.join(PREDICTIONS_BASE_DIR, loss_type, str(seed), "raw_predictions_for_calibration.npz")
        if not os.path.exists(file_path):
            print(f"  [Warning] Prediction file not found for seed {seed}: {file_path}")
            continue

        data = np.load(file_path)
        probabilities = data['probabilities']
        true_labels = data['labels']

        all_probas.append(probabilities)
        all_labels.append(true_labels)

        true_labels_one_hot = np.eye(probabilities.shape[1])[true_labels]
        brier = brier_score_loss(true_labels_one_hot.ravel(), probabilities.ravel())
        brier_scores.append(brier)

        ece = calculate_ece(probabilities, true_labels)
        eces.append(ece)

        print(f"  Seed {seed}: Brier Score = {brier:.4f}, ECE = {ece:.4f}")

    if all_probas:
        aggregated_probas = np.concatenate(all_probas, axis=0)
        aggregated_labels = np.concatenate(all_labels, axis=0)
    else:
        aggregated_probas, aggregated_labels = None, None

    return brier_scores, eces, aggregated_probas, aggregated_labels


def plot_calibration_curves(calibration_data, file_name="calibration_curves.png"):
    """
    Plots calibration curves for multiple models on a single figure.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for model_name, data in calibration_data.items():
        if data['probas'] is None:
            print(f"Skipping plot for {model_name} due to missing data.")
            continue

        confidences = np.max(data['probas'], axis=1)

        # --- THIS IS THE FIX ---
        # 1. Get the model's predicted class
        predictions = np.argmax(data['probas'], axis=1)
        # 2. Create a binary array: True if the prediction was correct, False otherwise
        correctness = (predictions == data['labels'])
        # 3. Pass this 'correctness' array to the function instead of the multiclass labels
        fraction_of_positives, mean_predicted_value = calibration_curve(
            correctness, confidences, n_bins=10, strategy='uniform'
        )

        ax.plot(mean_predicted_value, fraction_of_positives, "o-",
                label=f"{model_name} (ECE={data['avg_ece']:.4f})")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("Model Calibration Curves", fontsize=16)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"\n✅ Calibration curve plot saved to '{file_name}'")


def analyze_paired_tests(metrics_dict, model1_name, model2_name, metric_type):
    """Performs a Wilcoxon signed-rank test between two models for a given metric."""
    model1_scores = metrics_dict[model1_name][metric_type]
    model2_scores = metrics_dict[model2_name][metric_type]
    if len(model1_scores) != 5 or len(model2_scores) != 5: return
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
    if len(scores) < 2: return
    mean, std = np.mean(scores), np.std(scores)
    t_critical = stats.t.ppf(0.975, df=len(scores) - 1)
    margin_of_error = t_critical * (std / np.sqrt(len(scores)))
    ci_lower, ci_upper = mean - margin_of_error, mean + margin_of_error
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

    print(f"\n{'=' * 80}\n### CALIBRATION ANALYSIS ###\n{'=' * 80}")
    all_calibration_data = {}
    for model_name in models_to_analyze:
        brier_scores, eces, probas, labels = analyze_calibration(model_name, SEEDS)
        all_calibration_data[model_name] = {
            "Brier": brier_scores,
            "ECE": eces,
            "probas": probas,
            "labels": labels,
            "avg_ece": np.mean(eces) if eces else 0
        }
        if brier_scores:
            print(f"  > {model_name} Avg Brier: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")
            print(f"  > {model_name} Avg ECE:   {np.mean(eces):.4f} ± {np.std(eces):.4f}")

    print(f"\n{'=' * 80}\n### CONFIDENCE INTERVAL & STABILITY ANALYSIS ###\n{'=' * 80}")
    for model_name in models_to_analyze:
        print(f"\n--- Analysis for: {model_name} ---")
        for metric_name in ["Accuracy", "Macro_F1", "Demented_F1"]:
            analyze_confidence_intervals(final_metrics, model_name, metric_name)

    print(f"\n{'=' * 80}\n### PAIRED STATISTICAL TESTS (vs. FA-FL) ###\n{'=' * 80}")
    baselines = ["LDAM", "FocalLoss", "WCE"]
    for baseline_name in baselines:
        for metric_name in ["Accuracy", "Macro_F1", "Demented_F1"]:
            analyze_paired_tests(final_metrics, "FA_FL", baseline_name, metric_name)

    print(f"\n{'=' * 80}\n### GENERATING CALIBRATION CURVE PLOT ###\n{'=' * 80}")
    plot_calibration_curves(all_calibration_data)