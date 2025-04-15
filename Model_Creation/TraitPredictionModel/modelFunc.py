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

# ✅ Custom Gaussian NLL Loss
def gaussian_nll_loss(pred_mean, pred_logvar, target):
    precision = torch.exp(-pred_logvar)
    return torch.mean(precision * (target - pred_mean)**2 + pred_logvar)

# ✅ Laplace NLL Loss (Robust + Confidence-Aware)
def laplace_nll_loss(pred_mean, pred_logvar, target):
    scale = torch.exp(pred_logvar)  # predicted Laplace scale
    loss = torch.abs(target - pred_mean) / scale + pred_logvar
    return torch.mean(loss)

# ✅ Training Function
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, fileName, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (rgb_batch, dsm_batch, label_batch) in enumerate(train_loader):
            rgb_batch, dsm_batch, label_batch = rgb_batch.to(device), dsm_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()
            output = model(rgb_batch, dsm_batch)  # output: [B, 2]
            pred_mean = output[:, 0]
            pred_logvar = output[:, 1]
            loss = gaussian_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # ✅ Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_batch, dsm_batch, label_batch in val_loader:
                rgb_batch, dsm_batch, label_batch = rgb_batch.to(device), dsm_batch.to(device), label_batch.to(device)
                output = model(rgb_batch, dsm_batch)
                pred_mean = output[:, 0]
                pred_logvar = output[:, 1]
                loss = gaussian_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"✅ Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ✅ Save model
        torch.save(model.state_dict(), f"{fileName}{epoch+1}.pth")


# ✅ Full Training Function
def train_model_laplace(model, train_loader, val_loader, optimizer, scheduler, device, fileName,  num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for rgb_batch, dsm_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            rgb_batch = rgb_batch.to(device)
            dsm_batch = dsm_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            output = model(rgb_batch, dsm_batch)  # [B, 2]
            pred_mean = output[:, 0]
            pred_logvar = output[:, 1]

            loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ✅ Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_batch, dsm_batch, label_batch in val_loader:
                rgb_batch = rgb_batch.to(device)
                dsm_batch = dsm_batch.to(device)
                label_batch = label_batch.to(device)

                output = model(rgb_batch, dsm_batch)
                pred_mean = output[:, 0]
                pred_logvar = output[:, 1]

                loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ✅ Save Model
        torch.save(model.state_dict(), f"{fileName}{epoch+1}.pth")

def train_model_laplace_addoneinput(model, train_loader, val_loader, optimizer, scheduler, device, fileName, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for rgb_batch, dsm_batch, extra_input_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

        # ✅ Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_batch, dsm_batch, ear_weight_batch, label_batch in val_loader:
                rgb_batch = rgb_batch.to(device)
                dsm_batch = dsm_batch.to(device)
                ear_weight_batch = ear_weight_batch.to(device)
                label_batch = label_batch.to(device)

                output = model(rgb_batch, dsm_batch, ear_weight_batch)
                pred_mean = output[:, 0]
                pred_logvar = output[:, 1]

                loss = laplace_nll_loss(pred_mean, pred_logvar, label_batch.squeeze())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ✅ Save Model
        torch.save(model.state_dict(), f"{fileName}{epoch+1}.pth")

def train_model_laplace_addextrainput(model, train_loader, val_loader, optimizer, scheduler, device, fileName, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for rgb_batch, dsm_batch, extra_input_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

        # ✅ Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_batch, dsm_batch, extra_input_batch, label_batch in val_loader:
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

        print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ✅ Save model
        torch.save(model.state_dict(), f"{fileName}{epoch+1}.pth")

def setDevice():
    if torch.backends.mps.is_available():
        device = "mps"  # ✅ Use Apple Metal (Mac M1/M2)
        torch.set_default_tensor_type(torch.FloatTensor)
    elif torch.cuda.is_available():
        device = "cuda"  # ✅ Use NVIDIA CUDA (Windows RTX 4060)
    else:
        device = "cpu"  # ✅ Default to CPU if no GPU is available
    print(f"✅ Using device: {device}")
    return device

def setAndTrainModel(dataPath, traitName, model, savePath = "./",  num_epochs = 10, extraModelName = ""):
    '''
    set all data and train model
    dataPath, traitName, model, num_epochs
    '''
    # set model name and path to save model 
    modelName = model.__name__ + "_" + traitName

    if(extraModelName != ""):
        modelName += '_' + extraModelName

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
    train_model_laplace(modelName, train_loader, val_loader, optimizer, scheduler, device, saveModelPath,  num_epochs = num_epochs)

def setAndTrainModel_addoneinput(dataPath, extraInputName, traitName, model, savePath = "./",  num_epochs = 10):
    '''
    set all data and train model
    dataPath, traitName, model, num_epochs
    '''
    # set model name and path to save model 
    modelName = model.__name__ + "_" + traitName + "_extra" + extraInputName
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
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName=traitName, extra_input_cols=extraInputName)
    modelName = model().to(device)
    optimizer = optim.Adam(modelName.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # train and save model
    train_model_laplace_addoneinput(modelName, train_loader, val_loader, optimizer, scheduler, device, saveModelPath,  num_epochs = num_epochs)

def setAndTrainModel_addextrainput(dataPath, extraInputName, traitName, model, savePath="./", num_epochs=10, extraModelName = ""):
    """
    Set all data and train model with extra tabular input(s)
    extraInputName can be a single column (str) or list of columns
    """
    modelName = model.__name__ + "_" + traitName + "_extra_" + (
        extraInputName if isinstance(extraInputName, str) else "_".join(extraInputName)
    )

    if(extraModelName != ""):
        modelName += '_' + extraModelName

    saveModelPath = os.path.join(savePath, modelName)

    if not os.path.exists(savePath):
        print("❌ Path doesn't exist:", savePath)
        return

    print("✅ Save model to:", saveModelPath)

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
        modelInstance, train_loader, val_loader, optimizer, scheduler, device, saveModelPath, num_epochs=num_epochs
    )

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

def setAndTestModel(dataPath, traitName, model, modelPath):
    '''
    set data, device and test model
    '''
    # get data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName)
    
    # set device
    device = setDevice()

    # load model
    EfficientNetV2Model = model().to(device)
    if(device == "cuda"):
        EfficientNetV2Model.load_state_dict(torch.load(modelPath))
    else:
        EfficientNetV2Model.load_state_dict(torch.load(modelPath, map_location=torch.device("cpu")))
    EfficientNetV2Model.eval()

    print("traitName: ", traitName)
    print("model: ", model.__name__)

    # Run test
    df_results, r2, mae, rmse = new_test_model(EfficientNetV2Model, test_loader, device)

def new_test_model_with_extra_input(model, test_loader, device):
    model.eval()
    preds, stds, targets = [], [], []

    with torch.no_grad():
        for rgb_batch, dsm_batch, extra_input_batch, label_batch in tqdm(test_loader):
            # Move batches to the correct device
            rgb_batch, dsm_batch, extra_input_batch = rgb_batch.to(device), dsm_batch.to(device), extra_input_batch.to(device)
            
            # Run the model with RGB, DSM, and extra input
            output = model(rgb_batch, dsm_batch, extra_input_batch)  # [B, 2]

            # Extract prediction and log variance
            pred_mean = output[:, 0].cpu().numpy()
            pred_logvar = output[:, 1].cpu().numpy()
            pred_std = (torch.exp(0.5 * output[:, 1])).cpu().numpy()

            label_batch = label_batch.squeeze().cpu().numpy()

            preds.extend(pred_mean)
            stds.extend(pred_std)
            targets.extend(label_batch)

    # Calculate Metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = root_mean_squared_error(targets, preds)

    print(f"\n📊 Test Results:")
    print(f"✅ R² Score : {r2:.4f}")
    print(f"✅ MAE      : {mae:.4f}")
    print(f"✅ RMSE     : {rmse:.4f}")

    # Save predictions for analysis (optional)
    df = pd.DataFrame({
        "true": targets,
        "predicted": preds,
        "predicted_std": stds
    })
    df.to_csv("model_predictions_with_confidence.csv", index=False)

    return df, r2, mae, rmse


def setAndTestModel_with_extra_input(dataPath, traitName, model, modelPath, extraInputName):
    '''
    Set data, device, and test model with extra inputs.
    '''
    # Get data
    train_df, val_df, test_df = loadSplitData_no_leak(dataPath)
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName, extra_input_cols=extraInputName)
    
    # Set device (CPU, CUDA, MPS)
    device = setDevice()

    # Load model
    model_instance = model().to(device)
    if(device == "cuda"):
        model_instance.load_state_dict(torch.load(modelPath))
    else:
        model_instance.load_state_dict(torch.load(modelPath, map_location=torch.device("cpu")))
    model_instance.eval()

    print("traitName: ", traitName)
    print("model: ", model.__name__)

    # Run test with extra input
    df_results, r2, mae, rmse = new_test_model_with_extra_input(model_instance, test_loader, device)
