import os
import h5py
import numpy as np
import random


class HDF5DatasetWriter:
    def __init__(self, dims, output_path, batch_size=128, buffer_size=100):
        if os.path.exists(output_path):
            raise ValueError(output_path)
        self.db = h5py.File(output_path, "w")
        self.audios = self.db.create_dataset("audios", dims, dtype="float", compression="gzip",
                                             chunks=(batch_size, dims[1], dims[2], dims[3]))
        self.labels = self.db.create_dataset("labels", dims, dtype="float", compression="gzip",
                                             chunks=(batch_size, dims[1], dims[2], dims[3]))

        self.bufSize = buffer_size
        self.buffer = {"audios": [], "labels": []}
        self.idx = 0

    def add(self, audios, labels):
        self.buffer["audios"].extend(audios)
        self.buffer["labels"].extend(labels)
        if len(self.buffer["audios"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["audios"])
        self.audios[self.idx:i] = self.buffer["audios"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"audios": [], "labels": []}

    def close(self):
        if len(self.buffer["audios"]) > 0:
            self.flush()
        self.db.close()


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size):
        self.batchSize = batch_size
        self.db = h5py.File(db_path)
        self.numAudios = self.db["labels"].shape[0]

    def get_total_samples(self):
        return self.numAudios

    def generator(self):
        indexs = []
        for i in range(self.numAudios):
            indexs.append(i)
        random.shuffle(indexs)
        for i in np.arange(0, self.numAudios, self.batchSize):
            index = indexs[i]
            audios = self.db["audios"][index: index + self.batchSize]
            labels = self.db["labels"][index: index + self.batchSize]
            yield (audios, labels)

    def close(self):
        self.db.close()
