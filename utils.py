from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import os


def save_chunks(audio_path, save_dir, chunk_length_ms=1000):
    myaudio = AudioSegment.from_file(audio_path, "wav")
    chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec
    for i, chunk in tqdm(enumerate(chunks)):
        chunk_name = os.path.join(save_dir, "chunk{0}.wav".format(i))
        chunk.export(chunk_name, format="wav")
