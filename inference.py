# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time
import timm

from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from sklearn.metrics import f1_score, accuracy_score, classification_report
from utils.data_utils import get_loader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import csv

# grad_cam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision.models import resnet50
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
def plot(image, name, c, v, out_dir):
    c.reverse()
    v.reverse()
    fig = plt.figure(figsize=(18, 5))
    # print(c[4])
    plt.subplot(121)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image')
    plt.subplot(122)
    plt.barh(c,
        v, 
        color=['lightsteelblue'])
    plt.title('Prediction')
    # plt.title("Precision on ST Dogs", fontsize=20)
    plt.savefig(os.path.join(out_dir, name))
def Inference(args, model, image, classes, k=5):
    eval_losses = AverageMeter()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)['model']
        model.load_state_dict(checkpoint)
    logger.info("***** Running Inference *****")
    image = image.to(args.device)
    with torch.no_grad():
        model.eval()
        logit = model(image)
        smoid = nn.Sigmoid()
        smax = nn.Softmax(dim=1)
        # print(smax(smoid(logit)))
        # pred = torch.topk(F.softmax(logit, dim=1), 5, dim=1)
        # pred = torch.topk(smoid(logit), k)
        # print(smax(logit))
        pred = torch.topk(smax(logit), k, dim=1)

    topk_cls = []
    topk_value = []
    pred.indices.detach().cpu().numpy()[0]
    for i in range(k):
        prob = pred.values[0][i].item()
        topk_value.append((prob))
        topk_cls.append(classes[pred.indices[0][i]])
    return topk_cls, topk_value
def reshape_transform(tensor, height=37, width=37):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def visualization(model, input_tensor, rgb_img):
    target_layers = [model.transformer.encoder.layer[9].ffn_norm]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    image_att = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return image_att

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--dataset", choices=["cats", "dog", "oxford"], default="dog",
                        help="Which dataset.")
    parser.add_argument("--image_dir", default="./data/StanfordDogs/dog/Images/n02085620-Chihuahua/n02085620_275.jpg", type=str,
                        help="The image directory where input image exists.")
    parser.add_argument("--out_dir", default="./output", type=str,
                        help="The image directory where input image exists.")
    parser.add_argument("--image_name", default=None, type=str,
                        help="The name of output image file")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument('--checkpoint', type=str, default='./output/ST_dog_sigmoid_checkpoint.bin',
                        help="Model checkpoint")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.nprocs = torch.cuda.device_count()
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
    out_dir = args.out_dir
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
    model.to(args.device)
    # print(model)
    transform=transforms.Compose([  transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    
    image = Image.open(args.image_dir).convert('RGB')
    if args.image_name is not None:
        image_name = args.image_name
    else:
        image_name = args.image_dir.split('/')[-1].split('.')[0]
    rgb_img = image.resize((448, 448), Image.Resampling.BILINEAR)
    rgb_img = np.float32(rgb_img) / 255
    img = transform(rgb_img).view(1, 3, 448, 448)
    c, v = Inference(args, model, img, classes)
    plot(image, image_name, c, v, out_dir)
    # save image with att
    # image_att = visualization(model, img, rgb_img)
    # im = Image.fromarray(image_att)
    # im.save(os.path.join(out_dir, f"{image_name}_att.png"))
if __name__ == "__main__":
    main()


# n02109961-Eskimo_dog/n02109961_10699.jpg
# n02109961-Eskimo_dog/n02109961_4801.jpg
# n02109961-Eskimo_dog/n02109961_2832.jpg
# n02109961-Eskimo_dog/n02109961_1235.jpg
# n02109961-Eskimo_dog/n02109961_1413.jpg
# n02109961-Eskimo_dog/n02109961_19261.jpg
# n02109961-Eskimo_dog/n02109961_19358.jpg