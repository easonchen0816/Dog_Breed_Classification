# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from sklearn.metrics import f1_score, accuracy_score, classification_report
from utils.data_utils import get_loader
import matplotlib.pyplot as plt
import pandas as pd
import csv

logger = logging.getLogger(__name__)
def read_csv(path):
    l = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            l.append(row)
    return l
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def plot(cls_report_path):
    cls_name = []
    precision = []
    lines = read_csv(cls_report_path)
    for line in lines[1:-3]:
        cls_name.append(line[0])
        precision.append(int(float(line[1])*100))
    # precision.sort()
    # print(precision)
    plt.figure(figsize=(12, 7))
    plt.bar(cls_name,
        precision, 
        width=0.8, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue'])    
        #        'cornflowerblue', 
        #        'royalblue', 
        #        'midnightblue', 
        #        'navy', 
        #        'darkblue', 
        #        'mediumblue'
    plt.xticks(rotation='vertical', fontsize = 5)
    plt.title("Precision on ST Dogs", fontsize=20)
    plt.savefig('dog_breeds')
def validation(args, model, classes):
    _, test_loader = get_loader(args)
    eval_losses = AverageMeter()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)['model']
        model.load_state_dict(checkpoint)
    logger.info("***** Running Validation *****")

    with torch.no_grad():
        model.eval()
        all_preds, all_label = [], []
        epoch_iterator = tqdm(test_loader,
                            desc="Validating... (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True,
                            disable=args.local_rank not in [-1, 0])
        loss_fct = torch.nn.CrossEntropyLoss()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            # print(x.shape)
            with torch.no_grad():
                logits = model(x)

                eval_loss = loss_fct(logits, y)
                eval_loss = eval_loss.mean()
                eval_losses.update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        val_accuracy = accuracy.detach().cpu().numpy()
        target = all_label
        output = all_preds

        target_label = np.array([classes[target[i]] for i in range(len(target))], dtype=object)
        output_label = np.array([classes[output[i]] for i in range(len(output))], dtype=object)
        accuracy_model = accuracy_score(target_label, output_label)
        clf_report = classification_report(target_label, output_label, output_dict = False)
        f = open('oxford_report.txt', 'w')
        f.write('\n Accuracy\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, clf_report))
        f.close()
        # df = pd.DataFrame(clf_report).transpose()
        # df.to_csv('clf.csv')
        # plot(clf_report)
        logger.info("\n")
        logger.info("Validation Results")
        logger.info("Valid Loss: %2.5f" % eval_losses.avg)
        logger.info("Valid Accuracy: %2.5f" % val_accuracy)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default='ST_dog',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cats", "dog", "oxford"], default="dog",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/easonchen/TransFG/data/StanfordDogs')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/home/easonchen/TransFG/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--local_rank", type=int, default=-1,        
                        help="local_rank for distributed training on gpus")     #-1?
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument('--checkpoint', type=str, default='./output/ST_cat_checkpoint.bin',
                        help="Model checkpoint")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        #device = torch.device("cuda") 
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Set seed
    set_seed(args)
    oxford_classes = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    
    cats_classes = ['Calico', 'Tortoiseshell', 'Turkish Angora', 'Birman', 'Dilute Calico', 'Cornish Rex', 'Siberian', 'Munchkin', 'Turkish Van', 'Tuxedo', 'Scottish Fold', 'Oriental Short Hair', 'British Shorthair', 'Snowshoe', 'Maine Coon', 'Norwegian Forest Cat', 'Domestic Short Hair', 'Tiger', 'Torbie', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'American Shorthair', 'Burmese', 'American Bobtail', 'Persian', 'Russian Blue', 'Himalayan', 'Egyptian Mau', 'Sphynx - Hairless Cat', 'Dilute Tortoiseshell', 'Tabby', 'Siamese', 'Havana', 'Tonkinese', 'Ragdoll', 'Manx', 'Bombay', 'Balinese', 'Domestic Medium Hair', 'Bengal', 'Domestic Long Hair', 'Abyssinian']
    dogs_classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]
    # Model & Tokenizer Setup
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step
    if args.dataset == "dog":
        num_classes = 120
        classes = dogs_classes
    elif args.dataset == "cats":
        num_classes = 42
        classes = cats_classes
    elif args.dataset == "oxford":
        num_classes = 37
        classes = oxford_classes

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)

    #
    validation(args, model, classes)
    # plot('./clf.csv')

if __name__ == "__main__":
    main()
