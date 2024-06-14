from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.BrainImagingDataset import BrainImgDataset
from models.SimCLR import SimCLR



def load_datasets(hparams):

  if hparams.datatype == 'imaging':

    ####################
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.crop_scale_lower)
    hparams.transform = transform.__repr__()


    train_dataset = BrainImgDataset(
      transform=transform, table_dir=hparams.table_dir,
      root_dir=hparams.root_dir, modality_path=hparams.modality_path,
      split='train')

    val_dataset = BrainImgDataset(
      transform=transform, table_dir=hparams.table_dir,
      root_dir=hparams.root_dir, modality_path=hparams.modality_path, 
      split='val')
    return train_dataset, val_dataset




def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  # Create logdir based on WandB run name
  logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  
  model = SimCLR(hparams)
  
  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    callbacks.append(SSLOnlineEvaluator(z_dim = model.pooled_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal')))
  callbacks.append(ModelCheckpoint(filename='checkpoint_last_epoch_{epoch:02d}', dirpath=logdir, save_on_train_epoch_end=True, auto_insert_metric_name=False))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar)

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)