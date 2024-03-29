"Corpus splitting"


from dataclasses import dataclass

from omegaconf import MISSING, SI
from speechcorpusy import load_preset                                     # pyright: ignore [reportMissingTypeStubs]; bacause of library
from speechcorpusy.interface import ConfCorpus                            # pyright: ignore [reportMissingTypeStubs]; bacause of library
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId # pyright: ignore [reportMissingTypeStubs]
from configen import default                                              # pyright: ignore [reportMissingTypeStubs]


"""
(delele here when template is used)

[Design Notes - Corpus split]
    Split criteria is different in each corpus.
    Some audio corpus could need speaking-style split, but others would not need it.
    As a result, splitting logit is corpus-dependent, and splitting code become huge.
    For this reason, splitting logic is separated as a module here.

[Design Notes - Returns]
    Dataset do NOT have common Init interface because of its nature (e.g. no way to know even how many corpus it consumes).
    Splitted corpus is consumed by the datasets, so whole splitted output also do not have the interface.
    For this reason, splitter return/output is Dataset-dependet.
"""


CorpusItemIDs = tuple[AbstractCorpus, list[ItemId]]

@dataclass
class ConfCorpora:
    """Configuration of a `Corpora` instance."""
    root:   str        = MISSING             # Corpus data root
    n_val:  int        = MISSING             # The number of validation items, for corpus split
    n_test: int        = MISSING             # The number of test       items, for corpus split
    train:  ConfCorpus = default(ConfCorpus(
        root=SI("${..root}")))
    val:    ConfCorpus = default(ConfCorpus(
        root=SI("${..root}")))
    test:   ConfCorpus = default(ConfCorpus(
        root=SI("${..root}")))

def prepare_corpora(conf: ConfCorpora) -> tuple[CorpusItemIDs, CorpusItemIDs, CorpusItemIDs]:
    """Instantiate corpuses and split them for datasets.

    Returns - CorpusItems for train/val/test
    """

    # Instantiation
    ## No needs of content init. It is a duty of consumer (Dataset).
    corpus_train, corpus_val, corpus_test = load_preset(conf=conf.train), load_preset(conf=conf.val), load_preset(conf=conf.test)

    # Split
    ## e.g. Index-based split
    val_test = conf.n_val + conf.n_test
    items_train = corpus_train.get_identities()[            :-val_test]
    items_val   =   corpus_val.get_identities()[   -val_test:-conf.n_test]
    items_test  =  corpus_test.get_identities()[-conf.n_test:]
    ## e.g. Speaker-based split
    # if conf.name == "JVS":
    #     speakers_val = ["jvs095", "jvs096", "jvs098"]
    # else:
    #     speakers_val = []
    # items_val = list(filter(lambda item_id: item_id.speaker in speakers_val, corpus_val.get_identities()))

    # CorpusItem-nize
    corpus_items_train = (corpus_train, items_train)
    corpus_items_val   = (corpus_val,   items_val)
    corpus_items_test  = (corpus_test,  items_test)

    return corpus_items_train, corpus_items_val, corpus_items_test
