import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchaudio
import torchlibrosa as tl
from pydub import AudioSegment
import soundfile
from sklearn.model_selection import train_test_split

class Augmentations(nn.Module):
    def __init__(self, frequence_mask=2, time_mask=2):
        super().__init__()

        self.augs = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=frequence_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=frequence_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

    def forward(self, x):
        return self.augs(x)

def get_extractor(hop_length=256, win_length=1024, sample_rate=8000, n_mels=128):
        extractor = torch.nn.Sequential(
                    tl.Spectrogram( 
                        hop_length=hop_length,
                        win_length=win_length,
                    ), tl.LogmelFilterBank(
                        sr=sample_rate,
                        n_mels=n_mels,
                        is_log=False,
                    )) # (batch_size, channels, time_steps, mel_bins)
        return extractor


class WakeWordDataset(Dataset):
    def __init__(self, df: pd.DataFrame, duration_ms=4000, transforms = torchaudio.transforms.MFCC(), valid=False):
        self.df = df
        self.transforms = transforms

        win_length = 1024
        hop_length = 256
        n_mels = 128
        self.sample_rate = 8000
        self.duration_ms = duration_ms
        self.silent_audio = AudioSegment.silent(duration=duration_ms)
        extractor = get_extractor(hop_length, win_length, self.sample_rate, n_mels) # (batch_size, channels, time_steps, mel_bins)
        if valid:
            self.feature_extractor = extractor
        else:
            self.feature_extractor = nn.Sequential(
                extractor,
                Augmentations()
            )

    def pad_clip_audio(self, path):
        audio = self.silent_audio.overlay(AudioSegment.from_wav(path))
        audio = np.frombuffer(audio.set_frame_rate(self.sample_rate).set_channels(1)[0:self.duration_ms]._data, dtype='int16')
        return audio

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        file_path = self.df.loc[index, 'path']
        # data, sr = soundfile.read(file_path)
        data = self.pad_clip_audio(file_path)
        data = torch.Tensor(data.reshape(1, -1))
        data = self.feature_extractor(data)


        target = self.df.loc[index, 'wake']
        return data, torch.tensor([target], dtype=torch.float32)


def collate_function(batch):
    mfccs = []
    targets = []
    for mfcc, target in batch:
        mfccs.append(mfcc.squeeze(0).transpose(0,1))
        targets.append(target)

    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    mfccs = mfccs.transpose(1,2)
    return mfccs, torch.Tensor(targets)


def get_loaders(df: pd.DataFrame, batch_size=64):
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=12, stratify=df.wake, shuffle=True)
    train_df, test_df = train_df.reset_index(), test_df.reset_index()
    train_dataset = WakeWordDataset(train_df, )
    test_dataset = WakeWordDataset(test_df, valid=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=True)

    return train_loader, test_loader

