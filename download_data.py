import os
from scipy.io.wavfile import read

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


def walk_dir_and_write(dir1, dir2, f, emb_dir):
    for file in os.listdir(f"./data/{dir1}/{dir2}"):
        if '.wav' in file:
            text_path = f"./data/{dir1}/{dir2}/{file.split('.')[0] + '.txt'}"
            if os.path.exists(text_path):
                with open(text_path, 'r') as txt:
                    text = txt.read().strip()
                    _, data = read(f"./data/{dir1}/{dir2}/{file}")
                    if len(text) > 15 and len(data) != 0:
                        f.write(f"/data/vogorjachko/data/{dir1}/{dir2}/{file}|{text}|{emb_dir}/embeds/{dir1}_{dir2}_{file}.npy\n")


if __name__ == '__main__':
#     os.system("mkdir data")
#     for link in links:
#         os.system(f"wget {link}")
#         os.system(f"unzip {link.split('/')[-1]} -d data")
#         os.system(f"rm -rf {link.split('/')[-1]}")
#         break

    train = []
    val = []
    first_level = os.listdir('./data/')

    train = open('train.txt', 'w')
    emb_dir = "/data/vogorjachko"
    for dir1 in first_level[:-1]:
        for dir2 in os.listdir(f"./data/{dir1}/"):
            walk_dir_and_write(dir1, dir2, train, emb_dir)
    train.close()

    val = open('val.txt', 'w')
    for dir2 in os.listdir(f"./data/{first_level[-1]}/"):
        walk_dir_and_write(first_level[-1], dir2, val, emb_dir)
    val.close()