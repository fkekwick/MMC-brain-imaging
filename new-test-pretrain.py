import os 
import sys
import time
import random
from multiprocessing import Queue

import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from trainers.pretrain import pretrain
from trainers.evaluate import evaluate
from trainers.test import test
from trainers.generate_embeddings import generate_embeddings
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

import unittest
from hydra import compose, initialize
import multiprocessing as mp
import traceback

import run
import controller

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False




with initialize(version_base=None, config_path='./configs'):

    args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=imaging', 'pretrain=True'])
    self.run_process(controller.control, args)