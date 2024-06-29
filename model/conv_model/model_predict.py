import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay


def predict_model(model, testing_generator, true_labels, label_names, separate_labels=False, file=None, log_dir=""):
    """
    TODO: generator has true labels in the generator.y, use them.
    Args:
        model: Model instance
        testing_generator: Generator with audio data and true labels for prediction
        true_labels: True labels
        label_names: Label names, gotten from the mlb instance

    Returns: None
    """
    # Predict probabilities on the test set
    predicted_probs = model.predict(testing_generator, steps=len(testing_generator))

    if separate_labels:
        # Convert probabilities to binary predictions using a threshold
        predicted_labels = np.hstack([np.where(probs > 0.5, 1, 0) for probs in predicted_probs])
        true_labels = np.stack([testing_generator.y[:, i] for i in range(len(label_names))], axis=1)
    else:
        predicted_labels = [np.where(probs > 0.5, 1, 0) for probs in predicted_probs]
        true_labels = np.column_stack([testing_generator.y[:, i] for i in range(len(label_names))])

    print("\nClassification report:\n", file=file)
    print(classification_report(true_labels, predicted_labels, target_names=label_names), file=file)

    # Confusion matrices for each class
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)
    for i, class_name in enumerate(label_names):
        print(f"Confusion matrix for class {class_name}:", file=file)
        print(confusion_matrices[i],file=file)

    num_classes = len(label_names)
    num_cols = 4
    num_rows = int(np.ceil(num_classes / num_cols))

    # Setting up the figure, subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_classes * 2, 15))
    axes = axes.flatten()

    # Loop through all classes and plot the confusion matrix for each
    for i in range(num_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=["Absent", "Present"])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d', colorbar=False)
        axes[i].title.set_text(f'Class: {label_names[i]}')
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plot_file_path = os.path.join(log_dir, "confusion_matrices.png")
    plt.savefig(plot_file_path)
    plt.close()
    file.close()
    #plt.show()
