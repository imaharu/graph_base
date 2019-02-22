# my function
# Other
import os
import glob
import torch
import pickle
import argparse
from preprocessing import *
##### args #####
parser = argparse.ArgumentParser(description='Sequence to Sequence Model by using Pytorch')

parser.add_argument('--epoch', type=int, default=33)
parser.add_argument('--embed', type=int, default=128)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--batch', '-b', type=int, default=50)
parser.add_argument('--max_article_len', type=int, default=400)
parser.add_argument('--max_summary_len', type=int, default=100)

parser.add_argument('--generate_dir', '-g' ,type=str, default="val")
parser.add_argument('--result_file', '-r' ,type=str, default="rouge_result.txt")
parser.add_argument('--model_path', '-m' , type=str)
parser.add_argument('--save_dir', '-s' , type=str, default="train")
parser.add_argument('--cuda', '-c' , type=str, default="0")

parser.add_argument('--load_article_file', type=str, default="data/article.pt")
parser.add_argument('--load_summary_file', type=str, default="data/summary.pt")
parser.add_argument('--mode', type=str, default="dubug")
parser.add_argument('--none_bid', action='store_false')
parser.add_argument('--coverage', action='store_true')
parser.add_argument('--set_state', action='store_false')
args = parser.parse_args()
##### end #####

vocab_path = os.environ['cnn_vocab50000']
preprocess = Preprocess(args.max_article_len, args.max_summary_len)

source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)
source_size = len(source_dict)
target_size = len(target_dict)

if args.mode == "debug":
    article_data = preprocess.load("data/debug_article.pt")
    summary_data = preprocess.load("data/debug_summary.pt")

elif args.mode == "train":
    article_data = preprocess.load(args.load_article_file)
    summary_data = preprocess.load(args.load_summary_file)

elif args.mode == "val":
    generate_data = preprocess.load("data/val_article.pt")

hidden_size = args.hidden
embed_size = args.embed
max_epoch = args.epoch
batch_size = args.batch
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
