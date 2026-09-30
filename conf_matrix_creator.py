import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix values from your results
confusion_matrix = np.array([
    [980, 0, 20, 0],     # Mild Dementia
    [0, 98, 0, 0],       # Moderate Dementia
    [10, 0, 13300, 135], # Non Demented
    [0, 0, 140, 2605]    # Very mild Dementia
])

# Define class labels
classes = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

# Create figure and plot
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(confusion_matrix,
            annot=True,      # Show numbers in cells
            fmt='d',         # Use integer format
            cmap='Blues',    # Use blue color scheme
            xticklabels=classes,
            yticklabels=classes)

# Customize the plot
plt.title('Confusion Matrix (Test Set)', pad=20)
plt.xlabel('Predicted')
plt.ylabel('True')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()