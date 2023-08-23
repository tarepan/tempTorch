"""Separated for future library-nize."""

from typing import Any, TypeVar
from collections.abc import Callable
from hashlib import md5
import operator as ops
from functools import reduce
from pathlib import Path

from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId               # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.adress import dataset_adress                                 # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.access import generate_saver_loader, NTuple                  # pyright: ignore [reportMissingTypeStubs]


T = TypeVar("T", bound=NTuple)

def gen_ds_handlers(
        root: None | str,
        corpus: AbstractCorpus,
        items: list[ItemId],
        conf: Any,
        item_names: list[str],
        item_type: T,
    ) -> tuple[
        Callable[[ItemId, T], None],
        Callable[[ItemId],    T],
        Callable[[],          None],
        Callable[[],          bool],
        Path,
    ]:
    """Generate handlers for dataset.
    
    Returns:
        ...
        path_contents - Path of content root, exposed for raw path handling (e.g. inter-item statistics handling)
    """

    exp_specifier = md5(f"{items}{conf}".encode()).hexdigest()
    adress_archive, path_contents = dataset_adress(root, corpus.__class__.__name__, reduce(ops.add, item_names, ""), exp_specifier)
    save_item, load_item = generate_saver_loader(item_type, item_names, path_contents)

    def _save_archive() -> None:
        print("Archiving dataset contents...")
        save_archive(path_contents, adress_archive)
        print("Archived dataset contents.")

    def _load_archive() -> bool:
        contents_acquired = try_to_acquire_archive_contents(pull_from=adress_archive, extract_to=path_contents)
        if not contents_acquired:
            print(f"Dataset archive file is not found at {adress_archive}.")
        return contents_acquired

    return save_item, load_item, _save_archive, _load_archive, path_contents
