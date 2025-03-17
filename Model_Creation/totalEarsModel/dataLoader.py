import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WheatDataset(Dataset):
    def __init__(self, dataframe, rgb_col='rgbPath', dsm_col='dsmPath', label_col='totEarNum', 
                 height=256, width=512):
        """
        PyTorch Dataset for Wheat Ear Counting
        Args:
            dataframe (pd.DataFrame): DataFrame with file paths & labels.
            rgb_col (str): Column for RGB image paths.
            dsm_col (str): Column for DSM file paths.
            label_col (str): Column for label (totEarNum).
            height (int): Target height after resizing.
            width (int): Target width after resizing.
        """
        self.data = dataframe
        self.rgb_col = rgb_col
        self.dsm_col = dsm_col
        self.label_col = label_col
        self.height = height
        self.width = width
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.data)

    def get_transforms(self):
        """Define resizing and normalization steps."""
        return A.Compose([
            A.Resize(height=self.height, width=self.width),  # Resize both RGB & DSM
            A.Normalize(),  # Normalize RGB (0-1)
            ToTensorV2()
        ], additional_targets={'dsm': 'image'})  # Ensure DSM gets same resizing

    def normalize_dsm(self, dsm_array, fixed_min=200.0, fixed_max=230.0, nodata_value=-10000.0, fill_value=0.0):
        """Normalize DSM values to [0, 1] with NoData handling."""
        dsm_masked = np.ma.masked_equal(dsm_array, nodata_value)
        dsm_clipped = np.clip(dsm_masked, fixed_min, fixed_max)
        dsm_normalized = (dsm_clipped - fixed_min) / (fixed_max - fixed_min)
        return dsm_normalized.filled(fill_value).astype(np.float32)

    def __getitem__(self, idx):
        # ---- Load RGB ----
        rgb_path = self.data.loc[idx, self.rgb_col]
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb)  # Convert to NumPy (H, W, 3)

        # ---- Load DSM ----
        dsm_path = self.data.loc[idx, self.dsm_col]
        with rasterio.open(dsm_path) as src:
            dsm = src.read(1).astype(np.float32)  # Single-band DSM
        
        # Normalize DSM
        dsm = self.normalize_dsm(dsm)

        # ---- Apply resizing & normalization ----
        transformed = self.transform(image=rgb, dsm=dsm)
        rgb_tensor = transformed['image']  # (3, H, W)
        dsm_tensor = transformed['dsm'].unsqueeze(0)  # (1, H, W)

        # ---- Load label ----
        label = torch.tensor(self.data.loc[idx, self.label_col], dtype=torch.float32).unsqueeze(0)  # (1,)

        return rgb_tensor, dsm_tensor, label
