from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.BrainImagingDataset import BrainImgDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint


def test(hparams, wandb_logger=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  test_dataset = test_dataset = BrainImgDataset(table_dir=hparams.table_dir,root_dir=hparams.root_dir, modality_path=hparams.modality_path, split='test')
  drop = ((len(test_dataset)%hparams.batch_size)==1)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  hparams.dataset_length = len(test_loader)

  model = Evaluator(hparams)
  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
  trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)