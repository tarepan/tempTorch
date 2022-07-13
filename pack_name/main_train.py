"""Run training"""


import pytorch_lightning as pl
# import torchaudio

from .model import Model
from .data.datamodule import Data
from .config import load_conf
from lightlightning import train # pyright: ignore [reportMissingTypeStubs]


def main_train():
    """Train rnnms with cli arguments and the default dataset.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    # torchaudio.set_audio_backend("sox_io")
    model = Model(conf.model)
    model.train()
    datamodule = Data(conf.data)

    # Train
    train(model, conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
