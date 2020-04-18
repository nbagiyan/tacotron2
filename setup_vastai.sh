apt install vim screen gcc g++ libsndfile1-dev git wget zip

pip install matplotlib \
            tensorflow==1.15.2 \
            numpy \
            inflect \
            librosa \
            scipy \
            Unidecode \
            pillow \
            tensorboard \
            torch \
            tqdm \
            unidecode

git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

python download_data.py