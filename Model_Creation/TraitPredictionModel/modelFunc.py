import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataLoaderFunc import loadSplitData, createLoader, loadSplitData_no_leak
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# Custom Gaussian NLL Loss
def gaussian_nll_loss(pred_mean, pred_logvar, target):
    precision = torch.exp(-pred_logvar)
    return torch.mean(precision * (target - pred_mean)**2 + pred_logvar)

# Laplace NLL Loss (Robust + Confidence-Aware)
def laplace_nll_loss(pred_mean, pred_logvar, target):
    scale = torch.exp(pred_logvar)  # predicted Laplace scale
    loss = torch.abs(target - pred_mean) / scale + pred_logvar
    return torch.mean(loss)

def train_model_laplace(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler, 
    device, 
    fileName,  
    use_extra_input=False,
    max_epochs=50, 
    patience=7  # Number of epochs to wait before stopping
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            if use_extra_input:
                rgb_batch, dsm_batch, extra_input, label_batch, _ = batch
                extra_input = extra_input.to(device)
                output = model(rgb_batch.to(device), dsm_batch.to(device), extra_input)
            else:
                rgb_batch, dsm_batch, label_batch, _ = batch
                output = model(rgb_batch.to(device), dsm_batch.to(device))

            label_batch = label_batch.to(device)

            pred_mean = output[:, 0]
            pred_logvar = output[:, 1]
            loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_extra_input:
                    rgb_batch, dsm_batch, extra_input, label_batch, _ = batch
                    extra_input = extra_input.to(device)
                    output = model(rgb_batch.to(device), dsm_batch.to(device), extra_input)
                else:
                    rgb_batch, dsm_batch, label_batch, _ = batch
                    output = model(rgb_batch.to(device), dsm_batch.to(device))

                label_batch = label_batch.to(device)
                pred_mean = output[:, 0]
                pred_logvar = output[:, 1]
                loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{fileName}.pth")
            print(f"✅ Best model updated and saved at Epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
            break

def train_model_laplace_addextrainput(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler, 
    device, 
    fileName, 
    max_epochs=50, 
    patience=7
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for rgb_batch, dsm_batch, extra_input_batch, label_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)
            extra_input_batch = extra_input_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            output = model(rgb_batch, dsm_batch, extra_input_batch)
            pred_mean = output[:, 0]
            pred_logvar = output[:, 1]

            loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_batch, dsm_batch, extra_input_batch, label_batch, _ in val_loader:
                rgb_batch = rgb_batch.to(device)
                dsm_batch = dsm_batch.to(device)
                extra_input_batch = extra_input_batch.to(device)
                label_batch = label_batch.to(device)

                output = model(rgb_batch, dsm_batch, extra_input_batch)
                pred_mean = output[:, 0]
                pred_logvar = output[:, 1]

                loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{fileName}.pth")
            print(f"✅ Best model saved at Epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
            break


def setDevice():
    if torch.backends.mps.is_available():
        device = "mps"  # Use Apple Metal (Mac M1/M2)
        torch.set_default_tensor_type(torch.FloatTensor)
    elif torch.cuda.is_available():
        device = "cuda"  # Use NVIDIA CUDA (Windows RTX 4060)
    else:
        device = "cpu"  # Default to CPU if no GPU is available
    print(f"Using device: {device}")
    return device

def setAndTrainModel(dataPath, traitName, model,  savePath = "./", extraName = "none"):
    '''
    set all data and train model
    dataPath, traitName, model, num_epochs
    '''

    # set model name and path to save model 
    # modelName = "M-" + traitName + "_B-" + model.__name__ + "_D-" + dataPath.split("/")[-1].split(".")[0] + "_"

    modelName = traitName + "_" + model.__name__
    if(extraName != "none"):
        modelName = modelName + "_" + extraName

    print(modelName)
    saveModelPath = savePath + "/" + modelName

    # check if path to save exist
    if(not os.path.exists(savePath)):
        print("Path doesn't exists.")
        return 
    
    print("Save model to: ", saveModelPath)

    # set device cpu/cuda/mps
    device = setDevice()

    # get train_loader, val_loader, test_loader from data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName)
    modelName = model().to(device)
    optimizer = optim.Adam(modelName.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # train and save model
    train_model_laplace(modelName, train_loader, val_loader, optimizer, scheduler, device, saveModelPath)

def setAndTrainModel_addextrainput(dataPath, extraInputName, traitName, model, savePath="./", extraName = "none"):
    """
    Set all data and train model with extra tabular input(s)
    extraInputName can be a single column (str) or list of columns
    """
    # modelName = "M-" + traitName + "_B-" + model.__name__ + "_D-" + dataPath.split("/")[-1].split(".")[0] + "-E-" + (
    #     extraInputName if isinstance(extraInputName, str) else "-".join(extraInputName)
    # ) + "_"

    modelName = traitName + "_" + model.__name__ + "_" + (
        extraInputName if isinstance(extraInputName, str) else "-".join(extraInputName)
    ) 
    if(extraName != "none"):
        modelName = modelName + "_" + extraName

    saveModelPath = os.path.join(savePath, modelName)

    if not os.path.exists(savePath):
        print("Path doesn't exist:", savePath)
        return

    print("Save model to:", saveModelPath)

    # Set device
    device = setDevice()

    # Load data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(
        train_df, val_df, test_df, traitName=traitName, extra_input_cols=extraInputName
    )

    # Initialize model
    modelInstance = model().to(device)
    optimizer = optim.Adam(modelInstance.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Train model
    train_model_laplace_addextrainput(
        modelInstance, train_loader, val_loader, optimizer, scheduler, device, saveModelPath
    )

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def test_model_with_scatter_plot_shapeConfidence(
    model, test_loader, device,
    output_csv="model_predictions_with_confidence.csv",
    plot_title="Predicted vs True (±95% CI)",
    save_path="scatter_plot_confidence.png"
):
    model.eval()
    preds, stds, targets, RGBpaths = [], [], [], []

    with torch.no_grad():
        for rgb_batch, dsm_batch, label_batch, RGBpaths_batch in tqdm(test_loader, desc="Testing"):
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)

            output = model(rgb_batch, dsm_batch)  # [B, 2]
            pred_mean = output[:, 0].cpu().numpy()
            pred_std = (torch.exp(0.5 * output[:, 1])).cpu().numpy()
            label_batch = label_batch.squeeze().cpu().numpy()

            preds.extend(pred_mean)
            stds.extend(pred_std)
            targets.extend(label_batch.flatten().tolist())
            RGBpaths.extend(RGBpaths_batch)

    # Metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = root_mean_squared_error(targets, preds)

    print(f"\nTest Results:")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # Save predictions
    df = pd.DataFrame({
        "true": targets,
        "predicted": preds,
        "predicted_std": stds
    })
    df.to_csv(output_csv, index=False)

    # Compute relative std as confidence ratio
    preds_np = np.array(preds)
    stds_np = np.array(stds)
    relative_std = stds_np / (np.abs(preds_np) + 1e-6)  # avoid division by zero

    # Confidence groupings by relative uncertainty
    very_high_conf, high_conf, mid_conf, low_conf = [], [], [], []

    for t, p, r in zip(targets, preds, relative_std):
        if r < 0.05:
            very_high_conf.append((t, p))
        elif r < 0.10:
            high_conf.append((t, p))
        elif r < 0.20:
            mid_conf.append((t, p))
        else:
            low_conf.append((t, p))

    # Helper function to plot each group
    def plot_group(data, marker, label, color):
        if not data:
            return
        t, p = zip(*data)
        plt.scatter(t, p, marker=marker, alpha=0.7, s=20, label=label, color=color)

    # Plotting
    plt.figure(figsize=(6, 6), dpi=150)
    plot_group(very_high_conf, 'o', 'Very High Confidence', 'lime')
    plot_group(high_conf, 's', 'High Confidence', 'skyblue')
    plot_group(mid_conf, '^', 'Mid Confidence', 'coral')
    plot_group(low_conf, 'x', 'Low Confidence', 'crimson')

    # Identity line
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Ideal: y = x')
    plt.title(plot_title, fontsize=12)
    plt.xlabel("True Value", fontsize=10)
    plt.ylabel("Predicted Value", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    return df, r2, mae, rmse

def test_model_extra_input_with_scatter_plot_shapeConfidence(
    model, test_loader, device,
    output_csv="model_predictions_with_confidence.csv",
    plot_title="Predicted vs True (±95% CI, with Extra Input)",
    save_path="scatter_plot_confidence_extra_input.png"
):
    model.eval()
    preds, stds, targets, RGBpaths = [], [], [], []

    with torch.no_grad():
        for rgb_batch, dsm_batch, extra_input_batch, label_batch, RGBpaths_batch in tqdm(test_loader, desc="Testing"):
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)
            extra_input_batch = extra_input_batch.to(device)

            output = model(rgb_batch, dsm_batch, extra_input_batch)  # [B, 2]
            pred_mean = output[:, 0].cpu().numpy()
            pred_std = (torch.exp(0.5 * output[:, 1])).cpu().numpy()
            label_batch = label_batch.squeeze().cpu().numpy()

            preds.extend(pred_mean)
            stds.extend(pred_std)
            targets.extend(label_batch.flatten().tolist())
            RGBpaths.extend(RGBpaths_batch)

    # Metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = root_mean_squared_error(targets, preds)

    print(f"\nTest Results:")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # Save predictions
    df = pd.DataFrame({
        "true": targets,
        "predicted": preds,
        "predicted_std": stds,
        "RGBpath": RGBpaths
    })
    df.to_csv(output_csv, index=False)

    # Compute relative std as confidence ratio
    preds_np = np.array(preds)
    stds_np = np.array(stds)
    relative_std = stds_np / (np.abs(preds_np) + 1e-6)  # avoid division by zero

    # Confidence groupings by relative uncertainty
    very_high_conf, high_conf, mid_conf, low_conf = [], [], [], []

    for t, p, r in zip(targets, preds, relative_std):
        if r < 0.05:
            very_high_conf.append((t, p))
        elif r < 0.10:
            high_conf.append((t, p))
        elif r < 0.20:
            mid_conf.append((t, p))
        else:
            low_conf.append((t, p))

    # Helper function to plot each group
    def plot_group(data, marker, label, color):
        if not data:
            return
        t, p = zip(*data)
        plt.scatter(t, p, marker=marker, alpha=0.7, s=20, label=label, color=color)

    # Plotting
    plt.figure(figsize=(6, 6), dpi=150)
    plot_group(very_high_conf, 'o', 'Very High Confidence', 'lime')
    plot_group(high_conf, 's', 'High Confidence', 'skyblue')
    plot_group(mid_conf, '^', 'Mid Confidence', 'coral')
    plot_group(low_conf, 'x', 'Low Confidence', 'crimson')

    # Identity line
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Ideal: y = x')
    plt.title(plot_title, fontsize=12)
    plt.xlabel("True Value", fontsize=10)
    plt.ylabel("Predicted Value", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    return df, r2, mae, rmse

def setAndTestPlotModel(dataPath, traitName, model, modelPath):
    '''
    set data, device and test model
    '''
    # get data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName)
    
    # set device
    device = setDevice()

    # load model
    loadedModel = model().to(device)
    if(device == "cuda"):
        loadedModel.load_state_dict(torch.load(modelPath))
    else:
        loadedModel.load_state_dict(torch.load(modelPath, map_location=torch.device("cpu")))
    loadedModel.eval()

    print("traitName: ", traitName)
    print("model: ", model.__name__)

    # Run test
    df_results, r2, mae, rmse = test_model_with_scatter_plot_shapeConfidence(loadedModel, test_loader, device, 
                                                                   output_csv= "./ModelTestResult/" + traitName + "_predictions_with_confidence.csv",
                                                                   plot_title= traitName + ": Predicted vs True Value",
                                                                   save_path= "./ModelTestResult/" + traitName + "_scatter_plot_confidence.png")

def setAndTestPlotModel_with_extra_input(dataPath, traitName, model, modelPath, extraInputName):
    '''
    Set data, device, and test model with extra inputs.
    extraInputName: list of extra input column names, e.g., ["earWeight", "time"]
    '''
    # Get data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(
        train_df, val_df, test_df, traitName, extra_input_cols=extraInputName
    )

    # Set device (CPU, CUDA, MPS)
    device = setDevice()

    # Load model
    model_instance = model().to(device)
    if device == "cuda":
        model_instance.load_state_dict(torch.load(modelPath))
    else:
        model_instance.load_state_dict(torch.load(modelPath, map_location=torch.device("cpu")))
    model_instance.eval()

    print("traitName:", traitName)
    print("model:", model.__name__)

    # Convert extra input list to a string for title
    extra_input_str = ", ".join(extraInputName) if isinstance(extraInputName, list) else str(extraInputName)

    # Run test with extra input
    df_results, r2, mae, rmse = test_model_extra_input_with_scatter_plot_shapeConfidence(
        model_instance,
        test_loader,
        device=device,
        output_csv=f"./ModelTestResult/{traitName}_predictions_with_confidence.csv",
        plot_title=f"{traitName}: Predicted vs True Value, Extra Input: {extra_input_str}",
        save_path=f"./ModelTestResult/{traitName}_scatter_plot_confidence.png"
    )

    return df_results, r2, mae, rmse

def testModelByDate(
    model, test_loader, device,
    output_csv="model_predictions_with_confidence.csv",
    plot_title="Performance by Date",
    save_path="metrics_by_date.png"
):
    model.eval()
    preds, stds, targets, RGBpaths = [], [], [], []

    with torch.no_grad():
        for rgb_batch, dsm_batch, label_batch, RGBpaths_batch in tqdm(test_loader, desc="Testing"):
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)

            output = model(rgb_batch, dsm_batch)  # [B, 2]
            pred_mean = output[:, 0].cpu().numpy()
            pred_std = (torch.exp(0.5 * output[:, 1])).cpu().numpy()
            label_batch = label_batch.squeeze().cpu().numpy()

            preds.extend(pred_mean)
            stds.extend(pred_std)
            targets.extend(label_batch.flatten().tolist())
            RGBpaths.extend(RGBpaths_batch)

    # Store metrics by date
    metrics_by_date = {}
    currentDate = RGBpaths[0].split('/')[-1].split("_")[1][:8]
    currentPreds, currentTargets, currentStds = [], [], []

    for i in range(len(preds)):
        Date = RGBpaths[i].split('/')[-1].split("_")[1][:8]

        if Date != currentDate:
            if currentTargets and currentPreds:
                r2 = r2_score(currentTargets, currentPreds)
                mae = mean_absolute_error(currentTargets, currentPreds)
                rmse = np.sqrt(mean_squared_error(currentTargets, currentPreds))
                print(f"\nTest Results:")
                print(f"Date     : {currentDate}")
                print(f"R² Score : {r2:.4f}")
                print(f"MAE      : {mae:.4f}")
                print(f"RMSE     : {rmse:.4f}")
                metrics_by_date[currentDate] = r2
            currentDate = Date
            currentPreds, currentTargets, currentStds = [], [], []

        currentPreds.append(preds[i])
        currentTargets.append(targets[i])
        currentStds.append(stds[i])

    # Last group
    if currentTargets and currentPreds:
        r2 = r2_score(currentTargets, currentPreds)
        mae = mean_absolute_error(currentTargets, currentPreds)
        rmse = np.sqrt(mean_squared_error(currentTargets, currentPreds))
        print(f"\nTest Results:")
        print(f"Date     : {currentDate}")
        print(f"R² Score : {r2:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"RMSE     : {rmse:.4f}")
        metrics_by_date[currentDate] = r2

    # Plotting only R² scores with labels
    dates = list(metrics_by_date.keys())
    r2s = list(metrics_by_date.values())

    plt.figure(figsize=(12, 6))
    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("R² Score")
    plt.plot(dates, r2s, label="R² Score", marker='o', color='blue')

    for i, (date, r2) in enumerate(zip(dates, r2s)):
        plt.text(i, r2 + 0.01, f"{r2:.2f}", ha='center', fontsize=8, rotation=0, color='black')

    plt.xticks(ticks=range(len(dates)), labels=dates, rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def setAndTestModelByDate(dataPath, traitName, model, modelPath):
    '''
    set data, device and test model
    '''
    # get data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName)
    
    # set device
    device = setDevice()

    # load model
    loadedModel = model().to(device)
    if(device == "cuda"):
        loadedModel.load_state_dict(torch.load(modelPath))
    else:
        loadedModel.load_state_dict(torch.load(modelPath, map_location=torch.device("cpu")))
    loadedModel.eval()

    print("traitName: ", traitName)
    print("model: ", model.__name__)

    # Run test
    testModelByDate(loadedModel, test_loader, device, 
                    output_csv= "./ModelTestResult/" + traitName + "_predictions_with_confidence.csv",
                    plot_title= traitName + ": Predicted vs True Value",
                    save_path= "./ModelTestResult/" + traitName + "_scatter_plot_confidence.png")