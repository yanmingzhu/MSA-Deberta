from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import numpy as np
import torch
from util import *

def eval_model(model, test_loader, device):
    model.eval()
    model.to(device)
    total_sample = 0
    acc_2 = 0
    acc_7 = 0
    all_pred = np.array([])
    all_label = np.array([])
    with torch.no_grad():
        for batch in test_loader:
            text = batch[0].to(device)
            vision = batch[1].to(device)
            audio = batch[2].to(device)
            label = batch[3].to(device)
            masks = batch[4].to(device)
            #print(f"text {text.shape}")

            pred = model(text, audio, vision, masks).squeeze()

            label = label.squeeze()

            all_pred = np.append(all_pred, pred.cpu().detach().numpy())
            all_label = np.append(all_label, label.cpu().detach().numpy())


    pred_nonzero = all_pred[all_label != 0]
    label_nonzero = all_label[all_label != 0]
    mae = np.mean(np.abs(pred_nonzero - label_nonzero))
    corr = np.corrcoef(pred_nonzero, label_nonzero)[0,1]

    binary_pred = pred_nonzero >= 0
    binary_label = label_nonzero >= 0
    acc = accuracy_score(binary_pred, binary_label)
    f1 = f1_score(binary_pred , binary_label, average="weighted")
    print(f"acc_2 = {acc}, MAE = {mae}, f1 = {f1}, corr={corr}")

    acc, f1, mae, corr = metrics(all_pred, all_label)
    print(f"acc_2 = {acc}, f1 = {f1}, MAE = {mae}, , corr={corr}")

    return acc, mae, f1, corr
