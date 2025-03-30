import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rgbdsmAlignment import resize_dsm_return_array, resize_rgb_return_array, normalize_dsm

class WheatEarDataset(Dataset):
    def __init__(self, dataframe, rgb_col='rgb', dsm_col='dsm', label_col='totWheatEars', 
                 height=256, width=512):
        """
        Dataset for RGB + DSM + wheat ear count.
        Args:
            dataframe (pd.DataFrame): CSV containing paths and labels.
            rgb_col (str): Column name for RGB image paths.
            dsm_col (str): Column name for DSM file paths.
            label_col (str): Column for wheat ear count.
            height (int): Target height for resizing.
            width (int): Target width for resizing.
        """
        self.data = dataframe
        self.rgb_col = rgb_col
        self.dsm_col = dsm_col
        self.label_col = label_col
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Load RGB image
        # rgb_path = "/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.rgb_col]
        rgb_path = "D:/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.rgb_col]
        
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb)  # (H, W, 3)

        # Load DSM
        # dsm_path = "/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.dsm_col]
        dsm_path = "D:/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.dsm_col]
        
        with rasterio.open(dsm_path) as src:
            dsm = src.read(1).astype(np.float32)

        rgb = resize_rgb_return_array(rgb_path, target_size=(512, 256))  # (256, 512, 3)
        rgb = rgb / 255.0  # Normalize RGB to [0, 1]
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # (3, 256, 512)

        dsm = resize_dsm_return_array(dsm_path, target_width=512, target_height=256)
        dsm = normalize_dsm(dsm)  # normalize dsm
        dsm_tensor = torch.tensor(dsm, dtype=torch.float32).unsqueeze(0)  # (1, 256, 512)

         # ---- Load label ----
        label = torch.tensor([self.data.loc[idx, self.label_col]], dtype=torch.float32)

        return rgb_tensor, dsm_tensor, label
