from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import numpy as np

def metrics(all_pred, all_label):
    pred_nonzero = all_pred[all_label != 0]
    label_nonzero = all_label[all_label != 0]
    mae = np.mean(np.abs(pred_nonzero - label_nonzero))
    corr = np.corrcoef(pred_nonzero, label_nonzero)[0,1]

    binary_pred = pred_nonzero >= 0
    binary_label = label_nonzero >= 0
    acc = accuracy_score(binary_pred, binary_label)
    f1 = f1_score(binary_pred , binary_label, average="weighted")

    return acc, f1, mae, corr
 