# Dog Breed Classification

This repo contains the process of Dog Breed Classification

## Environments

The codebase is developed with Python 3.8.13. Install requirements as follows:

```bash
pip install -r requirements.txt
```

## Process

1. [Train](#train)
2. [Evaluate](#evaluate)
3. [Inference](#inference)

## Train

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 train.py --dataset oxford --data_root /RAID/eason --split overlap --num_steps 10000 --fp16 --name ST_oxford --train_batch_size 8
```

## Evaluate

```bash
python validation.py --dataset oxford --data_root /RAID/eason
```


## Inference

```bash
python inference.py --image_dir /home/easonchen/TransFG/data/test/362004.jpg --image_name shiba  --dataset oxford --
out_dir oxford_out --checkpoint /home/easonchen/TransFG/checkpoint/ST_oxford_checkpoint.bin
```

