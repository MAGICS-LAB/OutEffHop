import logging
import json
import math
import os
import random
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from pprint import pformat


import numpy as np
import torch
import transformers
# from accelerate import Accelerator
# from accelerate.utils import set_seed
# from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
# from timm.utils import AverageMeter
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

logger = logging.getLogger("validate_clm")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
from cross_models.stanhop import STanHopNet

model = STanHopNet(
            100, 
            128, 
            24,
            6,
)
logger.info(model)