import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import multilabel_confusion_matrix


class ConfusionMatrix:
    def __init__(self, y_true, y_pred, label_names, threshold=0.5):
        """
        Initialize the ConfusionMatrix class.

        Args:
            y_true (np.ndarray): True labels, shape (num_samples, num_classes).
            y_pred (np.ndarray): Predicted probabilities, shape (num_samples, num_classes).
            label_names (list): List of label names.
            threshold (float): Threshold to convert probabilities to binary predictions.
        """
        self.y_true = y_true
        self.y_pred = (y_pred >= threshold).astype(int)
        self.label_names = label_names
        self.num_classes = len(label_names)
        self.cm = multilabel_confusion_matrix(self.y_true, self.y_pred)

    def _compute_confusion_matrix(self):
        """
        Compute the confusion matrix for multi-label classification using sklearn's multilabel_confusion_matrix.

        Returns:
            pd.DataFrame: Confusion matrices for each label as a Pandas DataFrame.
        """
        # Prepare a list to store confusion matrices for each class
        cm_list = []

        for i, label in enumerate(self.label_names):
            tn, fp, fn, tp = self.cm[i].ravel()
            cm_list.append({
                'Class': label,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            })

        # Convert to a Pandas DataFrame for better visualization
        cm_df = pd.DataFrame(cm_list).set_index('Class')

        return cm_df

    def plot(self, title="Confusion Matrix", figsize=(15, 10), cmap='Blues', file_name=None, show_plot=False):
        """
        Plot the confusion matrix using Seaborn's heatmap.

        Args:
            title (str): Title of the plot.
            figsize (tuple): Figure size.
            cmap (str): Colormap to use for the heatmap.
            file_name (None | str | Path): If not None save the heatmap to a file
            show_plot (bool) Should the plot be shown (set False when running a lot of training in a row)
        """
        cm_df = self._compute_confusion_matrix()

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=figsize)
        sns.heatmap(cm_df[['TP', 'FP', 'FN', 'TN']], annot=True, fmt='d', cmap=cmap, cbar=False, linewidths=.5,
                    linecolor='black')

        plt.title(title)
        plt.xlabel('Predicted Outcome')
        plt.ylabel('Actual Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name)
        if show_plot:
            plt.show()
        plt.close()

    def get_confusion_matrix(self):
        """
        Get the confusion matrix DataFrame.

        Returns:
            pd.DataFrame: Confusion matrix DataFrame.
        """
        return self.cm

    def __str__(self):
        """
        Return a string representation of the confusion matrix.
        """
        result = "\nConfusion Matrix:\n"
        for i, label in enumerate(self.label_names):
            tn, fp, fn, tp = self.cm[i].ravel()
            result += f"{label}:\n"
            result += f"[[TN: {tn}, FP: {fp}]\n [FN: {fn}, TP: {tp}]]\n\n"
        return result
