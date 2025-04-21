import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def gaussian_nll_loss(pred_mean, log_var, target):
    return 0.5 * torch.exp(-log_var) * (pred_mean - target) ** 2 + 0.5 * log_var


#  Meta Model Training Function

def train_meta_model_with_validation(model, train_loader, val_loader, optimizer, scheduler, device, save_path, epochs=20):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for input_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(input_batch)
            pred, log_var = output[:, 0:1], output[:, 1:2]

            loss = gaussian_nll_loss(pred, log_var, label_batch).mean()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        total_train_loss /= len(train_loader)

        #  Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_batch, label_batch in val_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device).unsqueeze(1)
                output = model(input_batch)
                pred, log_var = output[:, 0:1], output[:, 1:2]
                loss = gaussian_nll_loss(pred, log_var, label_batch).mean()
                total_val_loss += loss.item()

        total_val_loss /= len(val_loader)
        scheduler.step(total_val_loss)

        print(f" Epoch {epoch+1} | Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), f"{save_path}.pth")


def setAndTrainMetaModel(dataPath, target_col, model, savePath="./", num_epochs=350):
    model_name = f"{target_col}_{model.__name__}"
    save_model_path = os.path.join(savePath, model_name)
    print(f"Model name: {model_name}")
    print(f"Saving to: {save_model_path}")

    if not os.path.exists(savePath):
        print("Save path does not exist.")
        return

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load and preprocess data
    df = pd.read_csv(dataPath)

    # Drop all true_ columns EXCEPT the one we're predicting
    true_cols_to_drop = [col for col in df.columns if col.startswith("true_") and col != f"true_{target_col}"]

    # Also drop non-input columns like DataKey
    drop_cols = ["DataKey"] + true_cols_to_drop

    # Keep only predicted columns (mean and std)
    input_cols = [col for col in df.columns if col.startswith("predicted_")]
    inputs = df[input_cols].values.astype("float32")
    labels = df[f"true_{target_col}"].values.astype("float32")


    # Confirm shape and input validity
    if inputs.shape[1] % 2 != 0:
        raise ValueError(f"Input feature count = {inputs.shape[1]} is not even. Expected [trait_mean, trait_std] pairs.")

    num_traits = inputs.shape[1] // 2
    print(f"Inferred number of traits: {num_traits} → Input shape: {inputs.shape}")

    print("\nInput columns used for training:")
    for col in input_cols:
        print(" -", col)

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    train_tensor = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_tensor   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_tensor, batch_size=32, shuffle=False)

    # Model & Optimizer
    model_instance = model(num_traits=num_traits).to(device)
    optimizer = Adam(model_instance.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Start training
    train_meta_model_with_validation(
        model_instance,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        save_model_path,
        epochs=num_epochs
    )




def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def test_meta_model_with_plot(
    model, test_loader, device,
    output_csv="meta_model_predictions.csv",
    plot_title="Meta Model Prediction vs True (±95% CI)",
    save_path="meta_model_scatter_plot.png"
):
    model.eval()
    preds, stds, targets = [], [], []

    with torch.no_grad():
        for input_batch, label_batch in tqdm(test_loader, desc="Testing"):
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device).unsqueeze(1)

            output = model(input_batch)
            pred_mean = output[:, 0].cpu().numpy()
            pred_std = (torch.exp(0.5 * output[:, 1])).cpu().numpy()
            label_batch = label_batch.squeeze().cpu().numpy()

            preds.extend(pred_mean)
            stds.extend(pred_std)
            targets.extend(label_batch)

    # Metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = root_mean_squared_error(targets, preds)

    print(f"\nMeta Model Test Results:")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # Save predictions
    df = pd.DataFrame({
        "true": targets,
        "predicted": preds,
        "predicted_std": stds,
        "lower_95CI": np.array(preds) - 1.96 * np.array(stds),
        "upper_95CI": np.array(preds) + 1.96 * np.array(stds)
    })
    df.to_csv(output_csv, index=False)

    # Plot
    plt.figure(figsize=(6, 6), dpi=150)
    plt.errorbar(targets, preds, yerr=1.96 * np.array(stds), fmt='o',
                 ecolor='gray', alpha=0.5, markersize=3, capsize=2,
                 label='Prediction ±95% CI')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Ideal: y = x')
    plt.title(plot_title)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    return df, r2, mae, rmse


def setAndTestMetaModel(dataPath, target_col, model_class, model_path):
    df = pd.read_csv(dataPath)

    # Split using GroupShuffleSplit on DataKey
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["DataKey"]))

    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Drop all other true columns (except target)
    true_cols_to_drop = [col for col in test_df.columns if col.startswith("true_") and col != f"true_{target_col}"]
    drop_cols = ["DataKey"] + true_cols_to_drop
    test_df = test_df.drop(columns=drop_cols)

    # Extract input + target
    input_cols = [col for col in test_df.columns if col.startswith("predicted_")]
    inputs = test_df[input_cols].values.astype("float32")
    labels = test_df[f"true_{target_col}"].values.astype("float32")

    print("\nInput columns used for testing:")
    for col in input_cols:
        print(" -", col)

    test_tensor = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    test_loader = DataLoader(test_tensor, batch_size=32, shuffle=False)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model_instance = model_class(num_traits=inputs.shape[1] // 2).to(device)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    # Run test
    return test_meta_model_with_plot(
        model_instance,
        test_loader,
        device,
        output_csv=f"{target_col}_meta_results.csv",
        plot_title=f"Meta Model: {target_col}",
        save_path=f"{target_col}_meta_plot.png"
    )
