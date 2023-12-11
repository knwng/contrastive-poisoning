import os
import torch
import numpy as np
import pickle

cifar10_dir = "cifar10-byol-poisoned-cps"
poison_filename = 'cifar10_res18_byol_cps.pth'

cifar10_dir = f"/data1/{cifar10_dir}/cifar-10-batches-py/"
poison_root = '/data1/cs260d/contrastive-poisoning/pretrained_poisons/'

cifar10_filename_list = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]

delta_weight = 8. / 255

def load_dataset_batch(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data # batch_label, labels, data, filenames

def dump_dataset_batch(path: str, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_poison_delta(path: str) -> torch.Tensor:
    return torch.load(path, map_location='cpu') # (50000, 3, 32, 32)

def main():
    poison_delta = load_poison_delta(os.path.join(poison_root, poison_filename))
    poison_delta = poison_delta.reshape(poison_delta.shape[0], -1).numpy()
    poison_delta = delta_weight * np.clip(poison_delta, a_min=-1.0, a_max=1.0)

    for i in range(len(cifar10_filename_list)):
        print(f'Processing {cifar10_filename_list[i]}...')
        filepath = os.path.join(cifar10_dir, cifar10_filename_list[i])
        original_data = load_dataset_batch(filepath)
        data = original_data['data']
        data_block_size = data.shape[0]
        data = data / 255.
        data += poison_delta[(i * data_block_size):((i + 1) * data_block_size), :]
        data = np.clip(data, a_min=0.0, a_max=1.0) * 255
        data = data.astype(np.uint8)
        original_data['data'] = data
        dump_dataset_batch(filepath, original_data)

if __name__ == "__main__":
    main()
