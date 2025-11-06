from random import random
import soundfile as sf
import librosa
import torch
from torch.utils import data
import numpy as np
import random

NOISY_DATABASE_TRAIN = '/home/wangzq_lab/cse12211026/SEtrain/Dataset/origin/train_noisy'
NOISY_DATABASE_VALID = '/home/wangzq_lab/cse12211026/SEtrain/Dataset/origin/train_noisy'

class DNS3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=16000,
        length_in_seconds=1,
        num_data_tot=2,
        num_data_per_epoch=1,
        random_start_point=False,
        train=True
    ):
        if train:
            print("You are using this DNS3 training data:", NOISY_DATABASE_TRAIN)
        else:
            print("You are using this DNS3 validation data:", NOISY_DATABASE_VALID)
        self.noisy_database_train = sorted(librosa.util.find_files(NOISY_DATABASE_TRAIN, ext='wav'))[:num_data_tot]
        self.noisy_database_valid = sorted(librosa.util.find_files(NOISY_DATABASE_VALID, ext='wav'))
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        
    def sample_data_per_epoch(self):
        print(f"self.num_data_per_epoch: {self.num_data_per_epoch}")
        print(f"self.noisy_database_train: {self.noisy_database_train}")
        if self.num_data_per_epoch > len(self.noisy_database_train):
            print(f"[Warning] num_data_per_epoch ({self.num_data_per_epoch}) > available data ({len(self.noisy_database_train)}), using random.choices (with replacement).")
            self.noisy_data_train = random.choices(self.noisy_database_train, k=self.num_data_per_epoch)
        else:
            self.noisy_data_train = random.sample(self.noisy_database_train, self.num_data_per_epoch)
        print(f"self.noisy_data_train:{len(self.noisy_data_train)}")

    def __getitem__(self, idx):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        if self.random_start_point:
            Begin_S = int(np.random.uniform(0, 10 - self.length_in_seconds)) * self.fs
            # Begin_S = 0
            noisy, _ = sf.read(noisy_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32',start=Begin_S, stop=Begin_S + self.L)

        else:
            noisy, _ = sf.read(noisy_list[idx], dtype='float32',start= 0, stop = self.L) 
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32', start=0, stop=self.L)
        ### only for debug
        print(len(noisy_list))
        print("shape:")
        print(noisy.shape)
        print(clean.shape)
        ###
        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)


if __name__=='__main__':
    from tqdm import tqdm 
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train.yaml')

        
    train_dataset = DNS3Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = DNS3Dataset(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass

    for noisy, clean in tqdm(validation_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass
