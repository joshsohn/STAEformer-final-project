import os
import numpy as np
import sys
import torch
import yaml

path = os.getcwd()
sys.path.append(path)
from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer import STAEformer

file_path = 'saved_models/STAEformer-PEMS08-2023-12-04-19-25-52.pt'

# Load the model
loaded_model = torch.load(file_path)

model = STAEformer(num_nodes=170)
model.load_state_dict(loaded_model)

# Load the dataset
dataset = "PEMS08"
dataset = dataset.upper()
path = os.getcwd() 

parentdir = os.path.abspath(path)
data_path = parentdir + f'/data/{dataset}'
data = np.load(os.path.join(data_path, "data.npz"))["data"].astype(np.float32)
data = torch.tensor(data)
model_name = STAEformer.__name__

with open(f"model/{model_name}.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg = cfg[dataset]

trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
    )

print("input dim is ", model.input_dim)
print(model.input_proj(data[:, :, model.input_dim].float()).shape)
print(model.tod_embedding(data[:, :, 1].long()).shape)
print(model.dow_embedding(data[:, :, 2].long()).shape)
print(model.adaptive_embedding.shape)

# Initialize an empty tensor

# batch_size = cfg.get("batch_size", 64)
# in_steps = cfg.get("in_steps")
# num_nodes =cfg.get("num_nodes")
# model_dim =cfg.get("model_dim", 3)
# all_embeddings = torch.empty((batch_size, in_steps, num_nodes, model_dim))

# counter = 0
# batch_embeddings = []
# all_embeddings = []
# for x_batch, y_batch in trainset_loader:
#     counter += 1
#     if counter % 100 == 0:
#         print(torch.cat(batch_embeddings).shape)
#         a = torch.cat((torch.cat(batch_embeddings), torch.randn((2000, 12, 170, 152))))
#         # batch_embeddings = []

#     batch_size = x_batch.shape[0]

#     if model.tod_embedding_dim > 0:
#         tod = x_batch[..., 1]
#     if model.dow_embedding_dim > 0:
#         dow = x_batch[..., 2]
#     x_batch = x_batch[..., : model.input_dim]

#     x_batch = model.input_proj(x_batch)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
#     features = [x_batch]
#     if model.tod_embedding_dim > 0:
#         tod_emb = model.tod_embedding(
#             (tod * model.steps_per_day).long()
#         )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
#         features.append(tod_emb)
#     if model.dow_embedding_dim > 0:
#         dow_emb = model.dow_embedding(
#             dow.long()
#         )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
#         features.append(dow_emb)
#     # if model.spatial_embedding_dim > 0:
#     #     spatial_emb = model.node_emb.expand(
#     #         batch_size, model.in_steps, *model.node_emb.shape
#     #     )
#     #     features.append(spatial_emb)
#     if model.adaptive_embedding_dim > 0:
#         adp_emb = model.adaptive_embedding.expand(
#             size=(batch_size, *model.adaptive_embedding.shape)
#         )
#         features.append(adp_emb)
#         # print(adp_emb.shape)
#     batch_embeddings.append(torch.cat(features, dim=-1)) # (batch_size, in_steps, num_nodes, model_dim)

# print(torch.cat(batch_embeddings).shape)


