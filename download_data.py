import os
import numpy as np

from scipy.io.wavfile import read
from get_mel import get_mel
from hparams import create_hparams


links = [
    "https://www.dropbox.com/s/s3ahqeggg9muba2/wav_tts_part1.zip",
    "https://www.dropbox.com/s/q0809p68sbqtkvr/wav_tts_part2.zip",
    "https://www.dropbox.com/s/acd41gahowcbycf/wav_tts_part3.zip",
    "https://www.dropbox.com/s/q5qtpc9x5wcewab/wav_tts_part4.zip",
    "https://www.dropbox.com/s/uwcscnb8gvkx175/wav_tts_part5.zip",
    "https://www.dropbox.com/s/96rquckkerwvseo/wav_tts_part6.zip",
    # "https://www.dropbox.com/s/4gq4c334e6w59a5/wav_tts_part7.zip",
    # "https://www.dropbox.com/s/lonm42nyv0f8jse/wav_tts_part8.zip"
]


def walk_dir_and_write(dir1, dir2, f):
    for file in os.listdir(f"./data/{dir1}/{dir2}"):
        if '.wav' in file:
            text_path = f"./data/{dir1}/{dir2}/{file.split('.')[0] + '.txt'}"
            if os.path.exists(text_path):
                with open(text_path, 'r') as txt:
                    text = txt.read().strip()
                    _, data = read(f"./data/{dir1}/{dir2}/{file}")
                    if 4 < len(text.split()) < 25 and len(data) != 0:
                        mel = get_mel(hparams, f"./data/{dir1}/{dir2}/{file}")
                        os.system(f"rm ./data/{dir1}/{dir2}/{file}")
                        np.save(f"/data/{dir1}/{dir2}/{file}", mel.cpu().numpy())
                        f.write(f"/data/{dir1}/{dir2}/{file}|{text}\n")


if __name__ == '__main__':
    hparams = create_hparams()
    os.system("mkdir data")
    for link in links:
        os.system(f"wget {link}")
        os.system(f"unzip {link.split('/')[-1]} -d data")
        os.system(f"rm -rf {link.split('/')[-1]}")

    train = []
    val = []
    first_level = os.listdir('./data/')

    train = open('train.txt', 'w')
    for dir1 in first_level[:-1]:
        for dir2 in os.listdir(f"./data/{dir1}/"):
            walk_dir_and_write(dir1, dir2, train)
    train.close()

    val = open('val.txt', 'w')
    for dir2 in os.listdir(f"./data/{first_level[-1]}/"):
        walk_dir_and_write(first_level[-1], dir2, val)
    val.close()