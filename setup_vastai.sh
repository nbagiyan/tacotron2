apt install -y vim screen gcc g++ libsndfile1-dev git wget zip ffmpeg

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

git clone https://github.com/nbagiyan/tacotron2.git
cd tacotron2 && python download_data.py

# python -m multiproc train.py --output_directory=outdir_opentts --log_directory=logdir \
# --hparams=distributed_run=True,fp16_run=Falseour,training_files=train.txt,validation_files=val.txt,batch_size=32,iters_per_checkpoint=2500