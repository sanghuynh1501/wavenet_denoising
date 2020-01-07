import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from hdf5_data import HDF5DatasetWriter


def load_data(data_path, second, total=0):

    df = pd.read_csv(data_path, sep='\t')
    print(df.head(10))

    dataset = None
    if "train" in data_path:
        dataset = HDF5DatasetWriter(dims=[total, 1, 128, 219], output_path="train.hdf5",
                                    batch_size=16)
    else:
        dataset = HDF5DatasetWriter(dims=[total, 1, 128, 219], output_path="test.hdf5",
                                    batch_size=16)

    count = 0
    with tqdm(total=total) as pbar:
        for sentence, path in zip(df["sentence"], df["path"]):
            label = None
            for i in range(6):
                file_name = path.split(".")[0] + "_" + str(i) + ".wav"
                sample_rate, samples = wavfile.read("data/speech_data/" + file_name)

                if len(samples) <= second * sample_rate:
                    if len(samples.shape) == 2:
                        samples = np.mean(samples, -1)
                    s = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
                    log_s = librosa.power_to_db(s, ref=np.max)
                    mfcc = librosa.feature.mfcc(S=log_s, n_mfcc=128)
                    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                    full_mfcc = []
                    for mfcc in delta2_mfcc:
                        while len(mfcc) < 219:
                            mfcc = np.concatenate((mfcc, np.array([0])), axis=0)
                        full_mfcc.append(mfcc)
                    full_mfcc = np.array(full_mfcc).astype(np.float)
                    full_mfcc = np.expand_dims(full_mfcc, axis=0)
                    full_mfcc = np.expand_dims(full_mfcc, axis=0)
                    if i == 0:
                        label = full_mfcc
                    else:
                        audio = full_mfcc
                        dataset.add(audio, label)

                    count += 1

                if total > 0:
                    if count >= total:
                        break
            if total > 0:
                if count >= total:
                    break
            pbar.update(1)
    print("count ", count)


second = 7

# all file 325740
load_data("en/train.tsv", second, 10000)
load_data("en/test.tsv", second, 2000)
