from pathlib import Path
import argparse
from multiprocessing.pool import Pool 
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import os
import glob
from scipy.io.wavfile import read

def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
#     wav = np.load(wav_fpath)
    sampling_rate, wav = read(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)
    
 
def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, save_path: Path, n_processes: int):
#     wav_dir = synthesizer_root.joinpath("audio")
#     metadata_fpath = synthesizer_root.joinpath("train.txt")
#     assert wav_dir.exists() and metadata_fpath.exists()
    print("Path", synthesizer_root)
    wav_paths = glob.glob(str(synthesizer_root) + "/*/*/*.wav")
#     print(wav_paths)
    embed_dir = save_path.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)
    
    # Gather the input wave filepath and the target output embed filepath
#     with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
#         metadata = [line.split("|") for line in metadata_file]
#         #print(metadata)
#         """for m in metadata:
#             #print(m)
#             fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2]))]"""
    
    
    fpaths = [(m, embed_dir.joinpath("_".join(m.split("/")[-3:]) + ".npy")) for m in wav_paths]
    print(fpaths[0])
    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--synthesizer_root",default="d/SV2TTS/synthesizer/", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    
    parser.add_argument("--save_path",default="d/embeds/", type=Path, help=\
    "Path to save embeds")
        
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    create_embeddings(**vars(args))    
