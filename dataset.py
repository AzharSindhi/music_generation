import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import os


class FluteMusicDataset(Dataset):
    def __init__(self, audio_dir):
        super().__init__()

        self.audio_dir = audio_dir
        self.chunk_filenames = sorted(os.listdir(audio_dir))
        # Define the transformations
        sample_rate = 441000
        target_sample_rate = 16000
        max_length = 16000

        self.transforms = Compose([
            torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate),
            torchaudio.transforms.MelSpectrogram(),
            torchaudio.transforms.AmplitudeToDB(),
            torchaudio.transforms.PadTrim(max_length),
        ])

    def __len__(self):
        return len(self.chunk_filenames)

    def __getitem__(self, index):
        label = 0  # since we do not need label
        path = os.path.join(self.audio_dir, self.chunk_filenames[index])
        print(os.path.exists(path))
        waveform, sr = torchaudio.load(path)
        print(sr)
        if self.transforms:
            waveform = self.transforms(waveform)

        return waveform, label


if __name__ == "__main__":
    path = "./data/chunks"
    dataset = FluteMusicDataset(path)
    print(len(dataset))
    # return
    X, y = dataset[0]
    print(X.shape)
