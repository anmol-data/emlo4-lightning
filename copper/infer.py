from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.vision import StandardTransform
from torchvision import transforms as T
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple
import torch
import torchvision.transforms as transforms
from PIL import Image
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule, Trainer

from copper import utils

log = utils.get_pylogger(__name__)

def preprocess_image(image_path):
    
    transforms = T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    
    image = Image.open(image_path).convert("RGB")  # Open the image and convert it to RGB if needed
    image = transforms(image)  # Apply the transformation
    
    return image.unsqueeze(0)


@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[dict, dict]:

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)


    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": None,
        "trainer": trainer,
    }


    log.info("Starting testing!")
    #trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    
    model.load_from_checkpoint('/Users/anmolsrivastava/emlo/emlo4-lightning/outputs/2023-07-01/12-58-13/lightning_logs/version_0/checkpoints/epoch=0-step=313.ckpt')
    model.eval()
    image_path = "/Users/anmolsrivastava/emlo/emlo4-lightning/data/PetImages_split/infer/Cat/12.jpg"  # Replace with the path to your image
    image = preprocess_image(image_path)
    # for predictions use trainer.predict(...)
    #predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path='/Users/anmolsrivastava/emlo/emlo4-lightning/outputs/2023-07-01/12-58-13/lightning_logs/version_0/checkpoints/epoch=0-step=313.ckpt')
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    print(predicted_class)
    
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    infer(cfg)

if __name__ == "__main__":
    main()