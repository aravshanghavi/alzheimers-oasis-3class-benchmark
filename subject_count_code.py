import os
import re
from collections import Counter, defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

# ================================================================================
# SECTION 1: CONFIGURATION
# Minimal configuration needed for subject counting.
# ================================================================================
CONFIG = {
    "DATA_DIR": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data",
    "RANDOM_STATE": 2024,
    "TEST_SET_RATIO": 0.20,
    "VAL_SET_RATIO": 0.15,
    "ORIGINAL_CLASSES": ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"],
    "NEW_CLASSES": ["Non Demented", "Very Mild Demented", "Demented"],
}

# ================================================================================
# SECTION 2: CORE HELPER FUNCTION
# This is the same function from your original script to identify subjects.
# ================================================================================
def _prepare_subject_data(data_dir, classes):
    """Scans directories and groups all images by subject ID."""
    subject_pattern = re.compile(r'(OAS\d_\d{4}_MR\d)')
    subjects_to_images = defaultdict(list)
    subject_to_label = {}
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for file_name in os.listdir(class_dir):
            match = subject_pattern.match(file_name)
            if match:
                subject_id = match.group(1)
                subjects_to_images[subject_id].append(os.path.join(class_dir, file_name))
                if subject_id not in subject_to_label:
                    subject_to_label[subject_id] = label_idx
    return list(subjects_to_images.keys()), subject_to_label

# ================================================================================
# SECTION 3: MAIN EXECUTION
# ================================================================================
if __name__ == '__main__':
    print("===== COUNTING UNIQUE SUBJECTS PER CLASS AND SPLIT =====")

    # 1. Load raw subject data and their original labels (0-3)
    all_subjects, subject_to_original_label = _prepare_subject_data(CONFIG["DATA_DIR"], CONFIG["ORIGINAL_CLASSES"])

    # 2. Remap subjects to the new 3-class problem (0-2)
    class_mapping = {
        CONFIG["ORIGINAL_CLASSES"].index("Non Demented"): CONFIG["NEW_CLASSES"].index("Non Demented"),
        CONFIG["ORIGINAL_CLASSES"].index("Very mild Dementia"): CONFIG["NEW_CLASSES"].index("Very Mild Demented"),
        CONFIG["ORIGINAL_CLASSES"].index("Mild Dementia"): CONFIG["NEW_CLASSES"].index("Demented"),
        CONFIG["ORIGINAL_CLASSES"].index("Moderate Dementia"): CONFIG["NEW_CLASSES"].index("Demented")
    }
    subject_to_new_label = {s: class_mapping[li] for s, li in subject_to_original_label.items()}
    all_new_labels = [subject_to_new_label[s] for s in all_subjects]

    # 3. Perform the exact same subject-level train/val/test split as the main script
    dev_subjects, test_subjects, dev_labels, _ = train_test_split(
        all_subjects, all_new_labels,
        test_size=CONFIG["TEST_SET_RATIO"],
        random_state=CONFIG["RANDOM_STATE"],
        stratify=all_new_labels
    )
    val_split_ratio_adjusted = CONFIG["VAL_SET_RATIO"] / (1 - CONFIG["TEST_SET_RATIO"])
    train_subjects, val_subjects, _, _ = train_test_split(
        dev_subjects, dev_labels,
        test_size=val_split_ratio_adjusted,
        random_state=CONFIG["RANDOM_STATE"],
        stratify=dev_labels
    )

    # 4. Count the number of subjects in each class for each split
    split_counts = {}
    for split_name, subject_list in [("Train", train_subjects), ("Validation", val_subjects), ("Test", test_subjects)]:
        # Get the labels for the subjects in the current split
        labels_in_split = [subject_to_new_label[subj] for subj in subject_list]
        # Count the occurrences of each label (0, 1, 2)
        counts = Counter(labels_in_split)
        # Store the counts, ensuring all classes are present (even if count is 0)
        split_counts[split_name] = [counts.get(i, 0) for i in range(len(CONFIG["NEW_CLASSES"]))]

    # 5. Create and display the results in a clean pandas DataFrame
    df = pd.DataFrame(split_counts, index=CONFIG["NEW_CLASSES"])
    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)

    print("\n--- Final Subject Counts ---")
    print(df.to_string())