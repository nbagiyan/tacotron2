import os
import pandas as pd
from multiprocessing import Pool


def convert_to_wav(filename):
    in_file = f'../clips/{filename}'
    out_file = f'../mozilla_wavs/{filename.replace("mp3", "wav")}'
    os.system(
        f'ffmpeg -i {in_file} -ar 16000 {out_file}'
    )
    return out_file


if __name__ == '__main__':
    os.system("mkdir mozilla_wavs")
    train = pd.read_csv('../train.tsv', sep='\t')
    dev = pd.read_csv('../dev.tsv', sep='\t')
    test = pd.read_csv('../test.tsv', sep='\t')

    train = pd.concat([train, dev], ignore_index=True)

    with Pool(processes=4) as pool:
        result_train = pool.map(convert_to_wav, train['path'].values.tolist())
        result_test = pool.map(convert_to_wav, test['path'].values.tolist())

    train['wav_path'] = result_train
    test['wav_path'] = result_test

    train.to_csv('train_mozilla.txt', sep='|', columns=['wav_path', 'sentence'], index=False, header=False)
    test.to_csv('val_mozilla.txt', sep='|', columns=['wav_path', 'sentence'], index=False, header=False)
