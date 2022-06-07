# configuration for training the Uformer Model in the Metz Datacenter of CentraleSupelec
# author: Igor Augusto Oliveira

import argparse
import os
import yaml

# Custom modules
from unet import ImageAugmentation

# Generating test data
###################
n_test_images = 1000

# Custom image generator
image_generator = ImageAugmentation(base_image_path="data/dataset/base_png")

# Generate test images according to the naming convention of the Uformer
# training procedure
image_generator.generate_input_images(n_images=n_test_images,
                                      save_path="data/dataset/train",
                                      generate_input=True,
                                      print_progress=False,
                                      input_dir='input',
                                      gt_dir='groundtruth')

# Processing runtime arguments
###################
parser = argparse.ArgumentParser()
parser.add_argument("--training_config", help="training configuration file *.yml", type=str,
                    required=False, default=None)
parser.add_argument("--arch", help="Model architecture (Uformer, Unet)", 
                    type=str, required=False, default="Uformer")
parser.add_argument("--batch_size", help="Batch size", type=int,
                    required=False, default=32)
parser.add_argument("--gpu", help="GPU device numbers (e.g.: '0,1' or '0')",
                    type=str, required=False, default="0")
parser.add_argument("--train_ps", help="Patch size in pixels", type=int,
                    required=False, default=128)
parser.add_argument("--train_dir", help="Path to training set", type=str,
                    required=False, default="data/dataset/train")
parser.add_argument("--env", 
                    type=str, required=False, default="000")
parser.add_argument("--val_dir", help="Path to validation set", type=str,
                    required=False, default="data/daset/val")
parser.add_argument("--embed_dim", help="Embedding dimension", type=int,
                    required=False, default=32)
parser.add_argument("--warmup", 
                    action="store_true")
args = parser.parse_args()

opt = vars(args)
if args.training_config:
    args = yaml.load(open(args.training_config), Loader=yaml.FullLoader)
    opt.update(args)
args = opt

# Calling UFormer training procedure
###################
# NOTE: There is surely some better way to pass these arguments.
uformer_dir = "Uformer"
os.chdir(uformer_dir)
args_str = ""
for key in args.keys():
    args_str += "--" + key + " " + str(args[key]) + " "
os.system("python3 train.py " + args_str)