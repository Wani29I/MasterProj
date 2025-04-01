import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

# Get current working directory instead of __file__
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from dataClass import WheatEarDataset

def loadSplitData(dataPath):
    # Load dataset
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

def loadSplitData_no_leak(dataPath, group_col="DataKey", val_size=0.1, test_size=0.1):
    df = pd.read_csv(dataPath)

    # Split off validation + test first using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=42)
    groups = df[group_col]
    train_idx, temp_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Now split temp into val and test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    groups_temp = temp_df[group_col]
    val_idx, test_idx = next(gss2.split(temp_df, groups=groups_temp))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"✅ Safe Split → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def createLoader(train_df, val_df, test_df, traitName):
    # Create dataset instances for each split
    train_dataset = WheatEarDataset(train_df, label_col = traitName)
    val_dataset = WheatEarDataset(val_df, label_col = traitName)
    test_dataset = WheatEarDataset(test_df, label_col = traitName)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)  # No shuffle for validation
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # No shuffle for testing

    # Check sizes
    print(f"Train Batches: {len(train_loader)}, Validation Batches: {len(val_loader)}, Test Batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader