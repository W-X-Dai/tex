import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

try:
    data_df = pd.read_excel("AcromegalyFeatureSet.xlsx")
    print("Read AcromegalyFeatureSet.xlsx successfully.")
except FileNotFoundError:
    print("Error: AcromegalyFeatureSet.xlsx not found.")
    exit()
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()
    
if "GroundTruth" not in data_df.columns:
    print("Error: 'GroundTruth' column not found.")
    exit()
y = data_df["GroundTruth"]

non_feature_cols = ["SeqNum", "Gender", "GroundTruth"]
X = data_df.drop(columns=non_feature_cols, errors='ignore') 
d_features = X.shape[1]
n_samples = X.shape[0]

print(f"Data preparation complete: {n_samples} samples, {d_features} features")

cpp_data_df = X.copy()
cpp_data_df['GroundTruth'] = y
txt_filename = "acromegaly_prepared.txt"
cpp_data_df.to_csv(
    txt_filename, 
    sep=' ', 
    header=False, 
    index=False,
    float_format='%.8f'
)
print(f"Data saved to '{txt_filename}' (including {d_features} features + 1 label).")

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Sensitivity (Recall or TPR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (TNR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return accuracy, sensitivity, specificity
        
    except ValueError:
        # if y has only one class present
        print("Warning: Error calculating confusion matrix.")
        return np.nan, np.nan, np.nan

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

print("\n--- Calculating OOB (Out-of-Bag) ---")
try:
    rf.fit(X, y)
    print(f"RF OOB Accuracy: {rf.oob_score_:.4f}")

    positive_class_index = np.where(rf.classes_ == 1)[0][0]
    y_probs_oob = rf.oob_decision_function_[:, positive_class_index]
    
    nan_mask = np.isnan(y_probs_oob)
    if np.any(nan_mask):
        print(f"Warning: {np.sum(nan_mask)} samples were never 'out-of-bag'.")
        y_true_valid = y[~nan_mask]
        y_probs_oob_valid = y_probs_oob[~nan_mask]
    else:
        y_true_valid = y
        y_probs_oob_valid = y_probs_oob

    auc_oob = roc_auc_score(y_true_valid, y_probs_oob_valid)
    fpr_oob, tpr_oob, _ = roc_curve(y_true_valid, y_probs_oob_valid)
    print(f"RF (True OOB) AUC: {auc_oob:.4f}")

    acc_oob, sens_oob, spec_oob = calculate_metrics(y_true_valid, y_probs_oob_valid)
    print(f"OOB Metrics (Th=0.5): Acc={acc_oob:.4f}, Sens={sens_oob:.4f}, Spec={spec_oob:.4f}")

except (AttributeError, ValueError) as e:
    print(f"Failed to calculate OOB AUC: {e}")
    auc_oob, fpr_oob, tpr_oob = None, None, None


print("\n--- Calculating 10-fold CV ---")
y_probs_cv10 = cross_val_predict(rf, X, y, cv=10, method='predict_proba')[:, 1]
auc_cv10 = roc_auc_score(y, y_probs_cv10)
fpr_cv10, tpr_cv10, _ = roc_curve(y, y_probs_cv10)
print(f"RF (10-fold CV) AUC: {auc_cv10:.4f}")

acc_cv10, sens_cv10, spec_cv10 = calculate_metrics(y, y_probs_cv10)
print(f"10-fold CV Metrics (Th=0.5): Acc={acc_cv10:.4f}, Sens={sens_cv10:.4f}, Spec={spec_cv10:.4f}")


print(f"\n--- Calculating LOOCV ( {n_samples} samples )...")
loo = LeaveOneOut()
y_probs_loocv = cross_val_predict(rf, X, y, cv=loo, method='predict_proba')[:, 1]
auc_loocv = roc_auc_score(y, y_probs_loocv)
fpr_loocv, tpr_loocv, _ = roc_curve(y, y_probs_loocv)
print(f"RF (LOOCV) AUC: {auc_loocv:.4f}")

acc_loocv, sens_loocv, spec_loocv = calculate_metrics(y, y_probs_loocv)
print(f"LOOCV Metrics (Th=0.5): Acc={acc_loocv:.4f}, Sens={sens_loocv:.4f}, Spec={spec_loocv:.4f}")


print("\n--- Drawing ROC Curves ---")
plt.figure(figsize=(10, 8))

if auc_oob is not None:
    plt.plot(fpr_oob, tpr_oob, 
             label=f'RF (OOB AUC = {auc_oob:.4f})', 
             linewidth=2, linestyle='-')

plt.plot(fpr_cv10, tpr_cv10, 
         label=f'RF (10-fold CV AUC = {auc_cv10:.4f})', 
         linewidth=2, linestyle='--')

plt.plot(fpr_loocv, tpr_loocv, 
         label=f'RF (LOOCV AUC = {auc_loocv:.4f})', 
         linewidth=2, linestyle=':')

plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance (AUC = 0.50)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Random Forest ROC Curve Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()

output_image_file = 'roc_curve_rf_comparison.png'
plt.savefig(output_image_file)
print(f"ROC Curve Comparison saved as '{output_image_file}'.")

"""
Read AcromegalyFeatureSet.xlsx successfully.
Data preparation complete: 103 samples, 19 features
Data saved to 'acromegaly_prepared.txt' (including 19 features + 1 label).

--- Calculating OOB (Out-of-Bag) ---
RF OOB Accuracy: 0.8252
RF (True OOB) AUC: 0.9076
OOB Metrics (Th=0.5): Acc=0.8252, Sens=0.7317, Spec=0.8871

--- Calculating 10-fold CV ---
RF (10-fold CV) AUC: 0.9058
10-fold CV Metrics (Th=0.5): Acc=0.8738, Sens=0.8293, Spec=0.9032

--- Calculating LOOCV ( 103 samples )...
RF (LOOCV) AUC: 0.9103
LOOCV Metrics (Th=0.5): Acc=0.8544, Sens=0.8049, Spec=0.8871

--- Drawing ROC Curves ---
ROC Curve Comparison saved as 'roc_curve_rf_comparison.png'.
"""