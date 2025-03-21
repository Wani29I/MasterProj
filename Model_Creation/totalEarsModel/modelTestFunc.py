import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Predicted vs. True Labels")
        plt.grid(True)
        plt.tight_layout()
        plt.show()