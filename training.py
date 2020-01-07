import numpy as np
from tqdm import tqdm
from torch import nn
import torch
import random

from hdf5_data import HDF5DatasetGenerator
from model import AutoEncoder

batch_size = 128
learning_rate = 0.01
train_gen = HDF5DatasetGenerator(db_path="train.hdf5", batch_size=batch_size)
test_gen = HDF5DatasetGenerator(db_path="test.hdf5", batch_size=batch_size)

network = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(500):

    network.train()
    min_audios = 0
    max_audios = 0
    min_labels = 0
    max_labels = 0
    total_loss = 0
    total = 0
    with tqdm(total=train_gen.get_total_samples() / batch_size) as pbar:
        for audios, labels in train_gen.generator():
            indexs = np.random.permutation(audios.shape[0])
            audios = audios[indexs]
            labels = labels[indexs]
            if np.min(audios) != 0 and np.max(audios) != 0 and np.min(labels) != 0 and np.max(labels) != 0:
                audios = audios / 60
                labels = labels / 60
                audios = torch.from_numpy(audios).float().cuda()
                labels = torch.from_numpy(labels).float().cuda()

                optimizer.zero_grad()
                outputs = network.forward(audios)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += 1

            pbar.update(1)

    train_loss = total_loss / total

    network.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=test_gen.get_total_samples() / batch_size) as pbar:
            for audios, labels in test_gen.generator():
                if np.min(audios) != 0 and np.max(audios) != 0 and np.min(labels) != 0 and np.max(labels) != 0:
                    audios = audios / 60
                    labels = labels / 60
                    audios = torch.from_numpy(audios).float().cuda()
                    labels = torch.from_numpy(labels).float().cuda()
                    outputs = network.forward(audios)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    total += 1

    test_loss = total_loss / total

    torch.save(network.state_dict(), "model.pt")

    print("epoch: ", epoch, "train loss: ", train_loss, "test loss: ", test_loss)