import numpy as np
import os
import shutil

def perturb_dataset(dataset="PEMS08", noise_level=5.0):
    dataset = dataset.upper()
    path = os.getcwd() 

    parentdir = os.path.abspath(path)
    data_path = parentdir + f'/data/{dataset}'
    data = np.load(os.path.join(data_path, "data.npz"))['data']
    print(data.shape)

    noise = np.random.normal(0, noise_level, data.shape)
    perturbed_data = data + noise
    print(perturbed_data.shape)
    
    perturbed_data_path = parentdir + f'/data/{dataset}_PERTURBED'
    if not os.path.exists(perturbed_data_path):
        os.makedirs(perturbed_data_path)
    np.savez(perturbed_data_path + '/data.npz', data=perturbed_data)

    shutil.copy2(os.path.join(data_path, "index.npz"), os.path.join(perturbed_data_path, "index.npz"))

perturb_dataset("PEMS08", 5.0)