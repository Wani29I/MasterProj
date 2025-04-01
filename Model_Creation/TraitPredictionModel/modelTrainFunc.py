import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataLoaderFunc import loadSplitData, createLoader

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

def setAllTrainModel(dataPath, traitName, model, savePath = "./",  num_epochs = 10):
    '''
    set all data and train model
    dataPath, traitName, model, num_epochs
    '''
    # set model name and path to save model 
    modelName = model.__name__ + "_" + traitName
    saveModelPath = savePath + "/" + modelName

    # check if path to save exist
    if(not os.path.exists(savePath)):
        print("Path doesn't exists.")
        return 
    
    print("Save model to: ", saveModelPath)

    # set device cpu/cuda/mps
    device = setDevice()

    # get train_loader, val_loader, test_loader from data
    train_df, val_df, test_df = loadSplitData(dataPath)
    train_loader, val_loader, test_loader = createLoader(train_df, val_df, test_df, traitName)
    modelName = model().to(device)
    optimizer = optim.Adam(modelName.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # train and save model
    train_model_laplace(modelName, train_loader, val_loader, optimizer, scheduler, device, saveModelPath,  num_epochs = num_epochs)