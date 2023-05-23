# from pydub import AudioSegment
# from pydub.utils import make_chunks
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import os


class FluteMusicDataset(Dataset):
    def __init__(self, audio_dir):
        super().__init__()

        self.audio_dir = audio_dir
        self.chunk_filenames = sorted(os.listdir(audio_dir))

    def __len__(self):
        return len(self.chunk_filenames)

    def __getitem__(self, index):
        label = 0  # since we do not need label
        path = os.path.join(self.audio_dir, self.chunk_filenames[index])
        waveform, sr = torchaudio.load(path)
        return waveform, label


if __name__ == "__main__":
    path = "./data/chunks"
    dataset = FluteMusicDataset(path)
    print(len(dataset))
    # return
    X, y = dataset[0]
    print(X.shape)
