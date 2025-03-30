import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Get current working directory instead of __file__
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from dataClass import WheatEarDataset

def loadSplitData(dataPath):
    # Load dataset
    # df = pd.read_csv("/Users/ice/Desktop/MasterResearch/MasterProj/Model_Creation/totalEarsModel/RGB_DSM_totEarNum.csv")
    df = pd.read_csv(dataPath)

    # Train-Validation-Test Split (80%-10%-10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # ✅ Reset index after splitting
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print sizes
    print(f"Train Size: {len(train_df)}, Validation Size: {len(val_df)}, Test Size: {len(test_df)}")

    return train_df, val_df, test_df

def createLoader(train_df, val_df, test_df):
    # Create dataset instances for each split
    train_dataset = WheatEarDataset(train_df)
    val_dataset = WheatEarDataset(val_df)
    test_dataset = WheatEarDataset(test_df)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)  # No shuffle for validation
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # No shuffle for testing

    # Check sizes
    print(f"Train Batches: {len(train_loader)}, Validation Batches: {len(val_loader)}, Test Batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader