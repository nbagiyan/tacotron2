import os
import numpy as np

from scipy.io.wavfile import read
from get_mel import get_mel
from hparams import create_hparams
from tqdm.auto import tqdm


links = [
    "https://www.dropbox.com/s/s3ahqeggg9muba2/wav_tts_part1.zip",
    "https://www.dropbox.com/s/q0809p68sbqtkvr/wav_tts_part2.zip",
    "https://www.dropbox.com/s/acd41gahowcbycf/wav_tts_part3.zip",
    "https://www.dropbox.com/s/q5qtpc9x5wcewab/wav_tts_part4.zip",
    "https://www.dropbox.com/s/uwcscnb8gvkx175/wav_tts_part5.zip",
    "https://www.dropbox.com/s/96rquckkerwvseo/wav_tts_part6.zip",
    "https://www.dropbox.com/s/4qtrsaq1576yloq/ours.zip",
    # "https://www.dropbox.com/s/4gq4c334e6w59a5/wav_tts_part7.zip",
    # "https://www.dropbox.com/s/lonm42nyv0f8jse/wav_tts_part8.zip"
]


def walk_dir_and_write(dir1, dir2, f):
    for file in os.listdir(f"./data/{dir1}/{dir2}"):
        if '.wav' in file:
            text_path = f"./data/{dir1}/{dir2}/{file.split('.')[0] + '.txt'}"
            audio_path = f"./data/{dir1}/{dir2}/{file}"
            if os.path.exists(text_path):
                with open(text_path, 'r') as txt:
                    text = txt.read().strip()
                    _, data = read(audio_path)
                    bytes_ = os.path.getsize(audio_path) - 44
                    mels = bytes_ / 512
                    if 4 < len(text.split()) < 25 and len(data) != 0 and mels < 800:
                        f.write(f"{audio_path}|{text}|./embedings/embeds/{dir1}/{dir2}/{file}.npy\n")


if __name__ == '__main__':
    hparams = create_hparams()
    os.system("mkdir data")
    for link in links:
        os.system(f"wget {link}")
        os.system(f"unzip {link.split('/')[-1]} -d data")
        os.system(f"rm -rf {link.split('/')[-1]}")
    os.system(f"wget https://www.dropbox.com/s/5ak061h8r7ylzzr/new_embs.zip")
    os.system(f"unzip new_embs.zip")
    train = []
    val = []
    first_level = os.listdir('./data/')

    train = open('train.txt', 'w')
    
    for dir1 in tqdm(first_level[:-1]):
        if dir1 == '.DS_Store':
            continue
        for dir2 in tqdm(os.listdir(f"./data/{dir1}/")):
            if dir2 == '.DS_Store':
                continue
            walk_dir_and_write(dir1, dir2, train)
    train.close()

    val = open('val.txt', 'w')
    for dir2 in os.listdir(f"./data/{first_level[-1]}/"):
        walk_dir_and_write(first_level[-1], dir2, val)
    val.close()

    os.system("mkdir outdir")
