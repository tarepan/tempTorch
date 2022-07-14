"""Datasets"""


from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from hashlib import md5

from torch import from_numpy, stack, load # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from omegaconf import MISSING, SI
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId               # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.adress import dataset_adress, generate_path_getter           # pyright: ignore [reportMissingTypeStubs]

from ..domain import FugaBatched, HogeBatched, HogeFugaBatch, LenFuga
from .domain import Hoge, HogeDatum, Fuga, FugaDatum, HogeFugaDatum
from .preprocessing import preprocess_hogefuga, ConfPreprocessing


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

[Design Notes - Corpus item path]
    Corpus instance has path-getter method.
    If we have multiple corpuses, we should distinguish the corpus that the item belongs to.
    If we pass paths as arguments, this problem disappear.
    For this reason, corpus_instance/selected_item_list/item_path are passed as arguments.

[Design Notes - Init interface]
    Dataset could consume multiple corpuses (corpus tuples), and the number is depends on project.
    For example, TTS will consumes single corpus, but voice conversion will consumes 'source' corpus and 'target' corpus.
    It means that init arguments are different between projects.
    For this reason, the Dataset do not have common init Inferface, it's up to you.
"""


CorpusItems = Tuple[AbstractCorpus, List[Tuple[ItemId, Path]]]


@dataclass
class ConfHogeFugaDataset:
    """Configuration of HogeFuga dataset.
    Args:
        adress_data_root - Root adress of data
        att1 - Attribute #1
    """
    adress_data_root: Optional[str] = MISSING
    attr1: int = MISSING
    preprocess: ConfPreprocessing = ConfPreprocessing(
        attr1=SI("${..attr1}"))

class HogeFugaDataset(Dataset[HogeFugaDatum]):
    """The Hoge/Fuga dataset from the corpus.
    """
    def __init__(self, conf: ConfHogeFugaDataset, items: CorpusItems):
        """
        Args:
            conf: The Configuration
            items: Corpus instance and filtered item information (ItemId/Path pair)
        """

        # Store parameters
        self._conf = conf
        self._corpus = items[0]
        self._items = items[1]

        # Calculate data path
        conf_specifier = f"{conf.attr1}{conf.preprocess}"
        item_specifier = f"{list(map(lambda item: item[0], self._items))}"
        exp_specifier = md5((conf_specifier+item_specifier).encode()).hexdigest()
        self._adress_archive, self._path_contents = dataset_adress(
            conf.adress_data_root, self._corpus.__class__.__name__, "HogeFuga", exp_specifier
        )
        self.get_path_hoge = generate_path_getter("hoge", self._path_contents)
        self.get_path_fuga = generate_path_getter("fuga", self._path_contents)

        # Deploy dataset contents
        ## Try to 'From pre-generated dataset archive'
        contents_acquired = try_to_acquire_archive_contents(self._adress_archive, self._path_contents)
        ## From scratch
        if not contents_acquired:
            print("Dataset archive file is not found.")
            self._generate_dataset_contents()

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and static preprocessing.
        """

        print("Generating new dataset...")

        # Lazy contents download
        self._corpus.get_contents()

        # Preprocessing
        for item in tqdm(self._items, desc="Preprocessing", unit="item"):
            preprocess_hogefuga(
                self._conf.preprocess, None,
                item[1], self.get_path_hoge(item[0]), self.get_path_fuga(item[0]),
            )

        print("Archiving new dataset...")
        save_archive(self._path_contents, self._adress_archive)
        print("Archived new dataset.")

        print("Generated new dataset.")

    def _load_datum(self, item_id: ItemId) -> HogeFugaDatum:
        """Data load with dynamic data modification.

        Load raw items, then dynamically modify it.
        e.g. Random clipping, Random noise addition, Masking
        """
        # Load
        hoge: Hoge = load(self.get_path_hoge(item_id))
        fuga: Fuga = load(self.get_path_fuga(item_id))

        # Modify
        ## :: (T,) -> (T=10, 1)
        hoge_datum: HogeDatum = np.expand_dims(hoge[:10], axis=-1) # pyright: ignore [reportUnknownMemberType]

        ## :: (T,) -> (T=10, 1)
        fuga_datum: FugaDatum = np.expand_dims(fuga[:10], axis=-1) # pyright: ignore [reportUnknownMemberType]

        return hoge_datum, fuga_datum

    def __getitem__(self, n: int) -> HogeFugaDatum:
        """Load the n-th datum from the dataset.

        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._items[n][0])

    def __len__(self) -> int:
        return len(self._items)

    def collate_fn(self, items: List[HogeFugaDatum]) -> HogeFugaBatch:
        """(API) datum-to-batch function."""

        hoge_batched: HogeBatched = stack([from_numpy(item[0]) for item in items])
        fuga_batched: FugaBatched = stack([from_numpy(item[1]) for item in items])
        len_fuga: LenFuga = [10, 10,]

        return hoge_batched, fuga_batched, len_fuga
