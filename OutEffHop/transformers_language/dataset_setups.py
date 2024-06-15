
from enum import auto

from quantization.utils import BaseEnumOptions


class DatasetSetups(BaseEnumOptions):
    wikitext_2 = auto()
    wikitext_103 = auto()
    bookcorpus_and_wiki = auto()
