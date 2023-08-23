"""Datasets"""

from dataclasses import dataclass

from torch.utils.data import Dataset
from tqdm import tqdm
from omegaconf import MISSING
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId # pyright: ignore [reportMissingTypeStubs]
from configen import default                                              # pyright: ignore [reportMissingTypeStubs]

from ..domain import HogeFugaBatch
from .domain import HogeFuga_, HogeFugaDatum
from .transform import ConfTransform, load_raw, preprocess, augment, collate
from .dataset_utils import gen_ds_handlers


"""
(delele here when template is used)

[Design Notes - Corpus instance]
    'Corpus in Dataset' enables corpus lazy evaluation, which enables corpus contents download skip.
    For this reason, `AbstractCorpus` instances are passed to the Dataset.

[Design Notes - Corpus item]
    Corpus split is logically separated from Dataset.
    If we get corpus items by corpus instance method call in the dataset, we should write split logic in the dataset.
    If we pass splitted items to the dataset, we can separate split logic from the dataset.
    For this reason, both corpus instance and selected item list are passed as arguments.

[Disign Notes - Responsibility]
    Data transformation itself is logical process (independent of implementation).
    In implementation/engineering, load/save/archiving etc... is indispensable.
    We could contain both data transform and enginerring in Dataset, but it can be separated.
    For this reason, Dataset has responsibility for only data handling, not data transform.
"""


@dataclass
class ConfHogeFugaDataset:
    """Configuration of a `HogeFugaDataset` instance."""
    root:         str | None    = MISSING                  # Dataset root adress
    save_archive: bool          = MISSING                  # Whether to save dataset archive
    transform:    ConfTransform = default(ConfTransform())

class HogeFugaDataset(Dataset[HogeFugaDatum]):
    """The Hoge/Fuga dataset.
    """
    def __init__(self, conf: ConfHogeFugaDataset, corpus: AbstractCorpus, item_ids: list[ItemId]):
        """
        Args:
            conf     - The Configuration
            corpus   - Corpus instance
            item_ids - Selected items in the corpus
        """

        # Configs
        item_names: list[str] = ["hoge", "fuga"]

        # Values and Handlers
        self._conf, self._corpus, self._item_ids = conf, corpus, item_ids
        self._save_item, self._load_item, self._save_archive, load_archive, _ = gen_ds_handlers(conf.root, corpus, item_ids, conf.transform.preprocess, item_names, HogeFuga_)

        # Deployment - Already-generated directory | extraction from pre-generated dataset archive | generation from scratch
        contents_loaded = load_archive()
        if not contents_loaded:
            self._generate_dataset_contents()

    def _generate_dataset_contents(self) -> None:
        """Generate dataset contents with corpus download, preprocessing and optional archiving."""

        print("Generating new dataset...")

        # Download - Lazy corpus contents download
        self._corpus.get_contents()

        # Preprocessing - Load/Transform/Save
        for item_id in tqdm(self._item_ids, desc="Preprocessing", unit="item"):
            raw = load_raw(self._conf.transform.load, self._corpus.get_item_path(item_id))
            item = preprocess(self._conf.transform.preprocess, raw)
            self._save_item(item_id, item)

        # Archiving - Optionally archive the dataset contents as an archive file for reuse
        if self._conf.save_archive:
            self._save_archive()

        print("Generated new dataset.")

    def __getitem__(self, n: int) -> HogeFugaDatum:
        """(API) Load the n-th datum from this dataset with augmentation (dynamic item tranformation).
        """
        return augment(self._conf.transform.augment, self._load_item(self._item_ids[n]))

    def __len__(self) -> int:
        return len(self._item_ids)

    def collate_fn(self, items: list[HogeFugaDatum]) -> HogeFugaBatch:
        """(API) datum-to-batch function (dyamic datum transformaion & dynamic batching)."""
        return collate(items)
