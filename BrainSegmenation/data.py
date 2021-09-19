from __future__ import annotations

import warnings, os, copy, time
from tqdm import tqdm
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torchvision


#@: Custom Image DataLoader CLass
class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, ...) -> None:
        ...



#@: class Data Analysis
class BrainAnalysis:
    def __init__(self) -> None:
        ...
    


#@: class Transformations
class BrainTransforms:
    def week_transforms(self) -> torchvision.transforms.Compose():
        ...
    

    def strong_transforms(self) -> torchvision.transforms.Compose():
        ...




#@: Driver Code
if __name__.__contains__('__main__'):
    pass