import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import argparse

def main(args):
    df = pd.read_csv(args.csv)


    # feature list
    feature_list = [
        'aleatoric_uncertainty',
        'epistemic_uncertainty',
        'predictive_uncertainty',
        'semantic_entropy',
        'semantic_entropy_entailment',
        'clip_score_variance',
        'lpips_diversity',
        'binary_entropy'
    ]

    y = df['label'].values

    for feature in feature_list:
        if feature not in df.columns:
            print(f"️ Feature '{feature}' not found in CSV. Skipping.")
            continue

        print(f"\nEvaluating Feature: {feature}")

        invert = feature in [
            'aleatoric_uncertainty',
            'epistemic_uncertainty',
            'predictive_uncertainty',
            'semantic_entropy_embedding',
            'semantic_entropy_entailment',
            'clip_score_variance',
            'binary_entropy'
        ]
        X = -df[feature].values.flatten() if invert else df[feature].values.flatten()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Threshold is the mean of training data
        threshold = np.mean(X_train)
        y_pred = (X_test > threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, X_test)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Threshold: {threshold:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        fpr, tpr, _ = roc_curve(y_test, X_test)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {feature}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        filename = f"roc_curve_{feature}.png"
        plt.savefig(filename)
        plt.close()
        print(f" ROC curve saved to '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to uncertainty_dataset.csv")
    args = parser.parse_args()

    main(args)
