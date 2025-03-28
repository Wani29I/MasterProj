import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

def new_test_model(model, test_loader, device):
    model.eval()
    preds, stds, targets = [], [], []

    with torch.no_grad():
        for rgb_batch, dsm_batch, label_batch in tqdm(test_loader):
            rgb_batch, dsm_batch = rgb_batch.to(device), dsm_batch.to(device)
            output = model(rgb_batch, dsm_batch)  # [B, 2]

            pred_mean = output[:, 0].cpu().numpy()
            pred_logvar = output[:, 1].cpu().numpy()
            pred_std = (torch.exp(0.5 * output[:, 1])).cpu().numpy()

            label_batch = label_batch.squeeze().cpu().numpy()

            preds.extend(pred_mean)
            stds.extend(pred_std)
            targets.extend(label_batch)

    # ✅ Metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = root_mean_squared_error(targets, preds)

    print(f"\n📊 Test Results:")
    print(f"✅ R² Score : {r2:.4f}")
    print(f"✅ MAE      : {mae:.4f}")
    print(f"✅ RMSE     : {rmse:.4f}")

    # ✅ Save predictions for analysis (optional)
    df = pd.DataFrame({
        "true": targets,
        "predicted": preds,
        "predicted_std": stds
    })
    df.to_csv("model_predictions_with_confidence.csv", index=False)

    return df, r2, mae, rmse

# ✅ Test the model on validation set
def test_model(model, test_loader):
    if torch.backends.mps.is_available():
        device = "mps"  # ✅ Use Apple Metal (Mac M1/M2)
        torch.set_default_tensor_type(torch.FloatTensor)
    elif torch.cuda.is_available():
        device = "cuda"  # ✅ Use NVIDIA CUDA (Windows RTX 4060)
    else:
        device = "cpu"  # ✅ Default to CPU if no GPU is available
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for rgb_batch, dsm_batch, label_batch in test_loader:
            rgb_batch, dsm_batch = rgb_batch.to(device), dsm_batch.to(device)
            outputs = model(rgb_batch, dsm_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(label_batch.cpu().numpy().flatten())

    return predictions, actuals

def evaluate_model(model, dataloader, device, plot_predictions=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb_batch, dsm_batch, labels in dataloader:
            rgb_batch, dsm_batch = rgb_batch.to(device), dsm_batch.to(device)
            labels = labels.to(device)

            outputs = model(rgb_batch, dsm_batch).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ✅ Evaluation Metrics
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)  # ✅ Manual RMSE
    r2 = r2_score(all_labels, all_preds)

    print("📊 Evaluation Results:")
    print(f"✅ MAE:   {mae:.2f}")
    print(f"✅ RMSE:  {rmse:.2f}")
    print(f"✅ R²:    {r2:.4f}")

    # ✅ Optional: Plot Predictions vs Ground Truth
    if plot_predictions:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_labels, all_preds, alpha=0.5)
        plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--')
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title("Predicted vs. True Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()