import pytorch_lightning as pl
import torch.nn as nn
import torch
import yaml
import argparse
from utils import read_data_file, load_images_and_labels
import pandas as pd
import sys
import os
import pdb


class Classifier(pl.LightningModule):






    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-c', default='configs/celebA_YSBBB_Classifier.yaml')


    # In parallel, creating dataloaders
    def prepare_data(self):
        txt_name, txt_path = "list_attr_celeba.txt", "../data/list_attr_celeba.txt"
        zip_file, zip_path = "img_align_celeba.zip", "../data/iimg_align_celeba.zip"

        assert (os.path.exists(txt_path), f"File {txt_name} not found")
        assert (os.path.exists(zip_file), f"File {zip_file} not found")








    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), 0.0001)
        return optim


    def training_step(self):
