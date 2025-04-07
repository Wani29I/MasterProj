import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rgbdsmAlignment import resize_dsm_return_array, resize_rgb_return_array, normalize_dsm

class WheatEarDataset(Dataset):
    def __init__(self, dataframe, key_col='DataKey', rgb_col='rgb', dsm_col='dsm', label_col='totWheatEars',
                 extra_input_cols =None, height=256, width=512):
        self.data = dataframe
        self.key_col = key_col
        self.rgb_col = rgb_col
        self.dsm_col = dsm_col
        self.label_col = label_col
        self.extra_input_cols = extra_input_cols  # Can be None
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Load RGB image
        # rgb_path = "D:/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.rgb_col]
        # rgb_path = "/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.rgb_col]
        rgb_path = "F:/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.rgb_col]
        rgb = resize_rgb_return_array(rgb_path, target_size=(512, 256)) / 255.0
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)

        # Load DSM
        # dsm_path = "D:/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.dsm_col]
        # dsm_path = "/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.dsm_col]
        dsm_path = "F:/ice-wheat/data/dataForProcess/mainData" + self.data.loc[idx, self.dsm_col]
        dsm = resize_dsm_return_array(dsm_path, target_width=512, target_height=256)
        dsm = normalize_dsm(dsm)
        dsm_tensor = torch.tensor(dsm, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor([self.data.loc[idx, self.label_col]], dtype=torch.float32)

        if self.extra_input_cols:
            if isinstance(self.extra_input_cols, list):
                extra_values = [self.data.loc[idx, col] for col in self.extra_input_cols]
                extra_input = torch.tensor(extra_values, dtype=torch.float32)
            else:
                extra_input = torch.tensor([self.data.loc[idx, self.extra_input_cols]], dtype=torch.float32)
            return rgb_tensor, dsm_tensor, extra_input, label
        else:
            return rgb_tensor, dsm_tensor, label


