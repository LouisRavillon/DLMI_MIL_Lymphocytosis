from functools import lru_cache
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from typing import Callable, Union
import torch


@lru_cache(maxsize=256)
def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)

class MILImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        training: bool = True,
        transform: Callable = None
    ):
        self.dataset = dataset
        self.training = training
        self.transform = transform

    @lru_cache(maxsize=4096)
    def __getitem__(self, index: int):
        row = self.dataset.loc[index]
        image_fp = row.tiles
        image = read_image(image_fp)
        age = row.age
        concentration = row.lymph_count
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.functional.to_tensor(image)
        if self.training:
            return index, image, np.array([row.label]).astype(float), torch.FloatTensor([age, concentration])
        else:
            return index, image, np.array([-1.0]), torch.FloatTensor([age, concentration])

    def __len__(self):
        return len(self.dataset)