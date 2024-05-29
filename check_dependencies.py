# check_dependencies.py
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score

print("All dependencies resolved successfully!")
