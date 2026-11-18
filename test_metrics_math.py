import numpy as np

def calculate_metrics(y_true, y_pred_prob, threshold=0.40, eps=1e-7):
    y_true_b = (y_true > 0.5).astype(np.uint8).flatten()
    y_pred_b = (y_pred_prob >= threshold).astype(np.uint8).flatten()
    
    tp = np.sum((y_true_b == 1) & (y_pred_b == 1))
    fp = np.sum((y_true_b == 0) & (y_pred_b == 1))
    fn = np.sum((y_true_b == 1) & (y_pred_b == 0))
    tn = np.sum((y_true_b == 0) & (y_pred_b == 0))
    
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_dice": float(f1_dice),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

# Test with simulated segmentation output
np.random.seed(42)
y_gt = np.zeros((10, 256, 256))
y_gt[:, 80:160, 80:160] = 1 # 10 samples with deficient core

# Predicted with slight shift
y_pred = np.zeros((10, 256, 256))
y_pred[:, 90:170, 90:170] = 0.85

res = calculate_metrics(y_gt, y_pred, threshold=0.5)
print("Diagnostic Metric Calculation Test:")
print(" - True Positives (TP):", f"{res['tp']:,}")
print(" - False Positives (FP):", f"{res['fp']:,}")
print(" - False Negatives (FN):", f"{res['fn']:,}")
print(" - True Negatives (TN):", f"{res['tn']:,}")
print(" - Precision:", f"{res['precision']:.4f}")
print(" - Recall:   ", f"{res['recall']:.4f}")
print(" - F1 (Dice):", f"{res['f1_dice']:.4f}")
print(" - IoU:      ", f"{res['iou']:.4f}")

# Verification of Dice to IoU mathematical identity: IoU = Dice / (2 - Dice)
expected_iou = res['f1_dice'] / (2.0 - res['f1_dice'])
diff = abs(res['iou'] - expected_iou)
assert diff < 1e-4, f"Math mismatch between Dice and IoU: {diff}"
print(" Mathematical identity IoU == Dice / (2 - Dice) verified: [PASS]")
