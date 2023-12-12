import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import yaml

from einops import rearrange
from matplotlib.colors import to_rgba
from sklearn.manifold import TSNE

path = os.getcwd()
sys.path.append(path)
from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer import STAEformer
from model.Spacetimeformer import Spacetimeformer
from lib.pca2 import pca_full_report

file_path = 'saved_models/Spacetimeformer-PEMS08-2023-12-10-01-54-41.pt'

# Load the model2
loaded_model = torch.load(file_path)

# model = Spacetimeformer(num_nodes=170)
model = Spacetimeformer(num_nodes=170)
model.load_state_dict(loaded_model)
print(model)

# Load the dataset
dataset = "PEMS08"
dataset = dataset.upper()
path = os.getcwd() 

parentdir = os.path.abspath(path)
data_path = parentdir + f'/data/{dataset}'
data = np.load(os.path.join(data_path, "data.npz"))["data"].astype(np.float32)[:, :, 0]
data = torch.tensor(data)
model_name = STAEformer.__name__

_, _, _, _, _, _, _, df_rank = pca_full_report(X=data, features_=np.arange(170))
top_n = 5
most_relevant_features = df_rank.loc[0:top_n, 'feature_'].to_list()
most_relevant_feature = most_relevant_features[0]
print(most_relevant_feature)

with open(f"model/{model_name}.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg = cfg[dataset]

trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
    )

# Initialize an empty tensor

# batch_size = cfg.get("batch_size", 64)
# in_steps = cfg.get("in_steps")
# num_nodes =cfg.get("num_nodes")
# model_dim =cfg.get("model_dim", 3)
# all_embeddings = torch.empty((batch_size, in_steps, num_nodes, model_dim))

for x_batch, y_batch in trainset_loader:
    print(x_batch[15, :, most_relevant_feature, :])
    y_predicted = model(x_batch)
    print(y_predicted.shape)

    y_predicted_single = y_predicted[15, :, most_relevant_feature, :]
    print(y_predicted_single)

    y_batch_single = y_batch[15, :, most_relevant_feature, :]
    print(y_batch_single)
    break