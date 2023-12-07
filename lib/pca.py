import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

dataset = "PEMS08"
dataset = dataset.upper()
# print("parent is ", parent)
path = os.getcwd() 

parentdir = os.path.abspath(path)
data_path = parentdir + f'/data/{dataset}'
data = np.load(os.path.join(data_path, "data.npz"))["data"].astype(np.float32)

print("data shape is ", data.shape)
# features = [0]
# if tod:
#     features.append(1)
# if dow:
#     features.append(2)
# # if dom:
# #     features.append(3)
# data = data[..., features]

# index = np.load(os.path.join(data_path, "index.npz"))

reshaped_data = data[:, :, 0]
print("reshaped data shape is ", reshaped_data.shape)
scaler = StandardScaler()
data_standardized = scaler.fit_transform(reshaped_data)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(data_standardized.T)

# Print the explained variance ratio for each principal component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# # Plot the cumulative explained variance
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('PCA: Cumulative Explained Variance')
# plt.show()

plt.figure(figsize=(10,10))
plt.scatter(principal_components[:,0],principal_components[:,1])
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()