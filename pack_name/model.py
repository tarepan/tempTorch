"""The model"""


from dataclasses import dataclass

import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from omegaconf import MISSING


from .domain import HogeFugaBatch
from .networks.network import Network, ConfNetwork


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_decay_rate: LR shaduler decay rate
        sched_decay_step: LR shaduler decay step
    """
    learning_rate: float = MISSING
    sched_decay_rate: float = MISSING
    sched_decay_step: int = MISSING

@dataclass
class ConfModel:
    """Configuration of the Model.
    """
    net: ConfNetwork = ConfNetwork()
    optim: ConfOptim = ConfOptim()

class Model(pl.LightningModule):
    """The model.
    """

    def __init__(self, conf: ConfModel):
        super().__init__()
        self.save_hyperparameters()
        self._conf = conf
        self._net = Network(conf.net)

    def forward(self, batch: HogeFugaBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Run inference toward a batch.
        """
        hoge, _, _ = batch

        # Inference :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        return self._net.generate(hoge)

    # Typing of PL step API is poor. It is typed as `(self, *args, **kwargs)`.
    def training_step(self, batch: HogeFugaBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Train the model with a batch.
        """

        hoge, fuga_gt, _ = batch

        # Forward :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        fuga_pred = self._net(hoge)

        # Loss
        loss = F.l1_loss(fuga_pred, fuga_gt)

        self.log('loss', loss) #type: ignore ; because of PyTorch-Lightning
        return {"loss": loss}

    def validation_step(self, batch: HogeFugaBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Validate the model with a batch.
        """

        i_pred, o_gt, _ = batch

        # Forward :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        o_pred_fwd = self._net(i_pred)

        # Inference :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        ## Usecase: Autoregressive model (`o_pred_fwd` for teacher-forcing, `o_pred_inf` for AR generation)
        # o_pred_inf = self.net.generate(i_pred)

        # Loss
        loss_fwd = F.l1_loss(o_pred_fwd, o_gt)

        # Logging
        ## Audio
        # # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)
        # #                                                      ::Tensor(1, L)
        # self.logger.experiment.add_audio(f"audio_{batch_idx}", o_pred_fwd, global_step=self.global_step, sample_rate=self.conf.sampling_rate)

        return {
            "val_loss": loss_fwd,
        }

    # def test_step(self, batch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Test a batch. If not provided, test_step == validation_step."""
    #     return anything_for_`test_epoch_end`

    def configure_optimizers(self): # type: ignore ; because of PyTorch-Lightning (no return typing, so inferred as Void)
        """(PL API) Set up a optimizer.
        """
        conf = self._conf.optim

        optim = Adam(self._net.parameters(), lr=conf.learning_rate)
        sched = {
            "scheduler": StepLR(optim, conf.sched_decay_step, conf.sched_decay_rate),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }

    # def predict_step(self, batch: HogeFugaBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Run prediction with a batch. If not provided, predict_step == forward."""
    #     return pred

    def preprocess(self):
        pass

    def sample(self):
        """Acquire sample input."""
        pass
        # wave, sr = librosa.load(librosa.example("libri2"), sr=self.conf.sampling_rate)
        # return wave, sr
