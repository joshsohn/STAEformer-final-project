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

pt_file = 'STAEformer-PEMS08-2023-12-04-19-25-52.pt'
file_path = 'saved_models/' + pt_file

# Load the model2
loaded_model = torch.load(file_path)

model = STAEformer(num_nodes=170)
# model = Spacetimeformer(num_nodes=170)
model.load_state_dict(loaded_model)

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
print(most_relevant_features)

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

counter = 0
num_batch = 1
embeddings = []
for x_batch, y_batch in trainset_loader:
    if counter == num_batch:
        break
    counter += 1

    batch_size = x_batch.shape[0]

    if model.tod_embedding_dim > 0:
        tod = x_batch[..., 1]
    if model.dow_embedding_dim > 0:
        dow = x_batch[..., 2]
    x_batch = x_batch[..., : model.input_dim]

    x_batch = model.input_proj(x_batch)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
    features = []
    # features = [x_batch]
    if model.tod_embedding_dim > 0:
        tod_emb = model.tod_embedding(
            (tod * model.steps_per_day).long()
        )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
        # features.append(tod_emb)
        # print(tod_emb.shape)
    if model.dow_embedding_dim > 0:
        dow_emb = model.dow_embedding(
            dow.long()
        )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
        # features.append(dow_emb)
        # print(dow_emb.shape)
    if model.adaptive_embedding_dim > 0:
        adp_emb = model.adaptive_embedding.expand(
            size=(batch_size, *model.adaptive_embedding.shape)
        )
        features.append(adp_emb)
        # print(adp_emb.shape)
    embeddings.append(torch.cat(features, dim=-1)) # (batch_size, in_steps, num_nodes, model_dim)

all_embeddings = torch.cat(embeddings)
print(all_embeddings.shape)

def generate_random_rgb():
    return (random.random(), random.random(), random.random())

# Generate 170 different RGB values
num_colors = 170
rgb_values = [generate_random_rgb() for _ in range(num_colors)]

# # Convert RGB values to Matplotlib color strings
# mpl_colors = [to_rgba(rgb) for rgb in rgb_values]

# Assuming X is your flattened data
flattened_embeddings = rearrange(all_embeddings.detach().numpy(), 'batch_size in_steps num_nodes model_dim -> (batch_size in_steps num_nodes) model_dim')
tsne = TSNE(n_components=2, perplexity=100)
print('tsne')
tsne_embeddings = tsne.fit_transform(flattened_embeddings)
print('tsne_embeddings')
unflattened = torch.reshape(torch.tensor(tsne_embeddings), (16*num_batch, 12, 170, 2))
# unflattened_embeddings = rearrange(tsne_embeddings, 'n model_dim -> (16 12 170) model_dim')
plt.figure(figsize=(30, 20))
for i in most_relevant_features:
    time_i = unflattened[:, :, i:i+1, :]
    
    flattened = rearrange(time_i, 'batch_size in_steps num_nodes model_dim -> (batch_size in_steps num_nodes) model_dim')
    
    print('flattened:', flattened.shape)

    plt.scatter(flattened[:, 0], flattened[:, 1], c=rgb_values[i], s=30)
    
# randn = torch.randn_like(torch.tensor(flattened_embeddings))

print('tsne')

# tsne_embeddings = tsne.fit_transform(randn)
print('tsne_embeddings')


plt.colorbar()
plt.title('t-SNE Plot along Spatial Axis')
plt.show()

plt.figure(figsize=(30, 20))
for i in range(170):
    time_i = unflattened[:, :, i:i+1, :]
    
    flattened = rearrange(time_i, 'batch_size in_steps num_nodes model_dim -> (batch_size in_steps num_nodes) model_dim')
    
    plt.scatter(flattened[:, 0], flattened[:, 1], c=rgb_values[i], s=30)
    
# randn = torch.randn_like(torch.tensor(flattened_embeddings))

print('tsne')

# tsne_embeddings = tsne.fit_transform(randn)
print('tsne_embeddings')


plt.colorbar()
plt.title('t-SNE Plot along Spatial Axis')
plt.show()