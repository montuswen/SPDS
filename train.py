import os
import time

import torch
from transformers import AutoTokenizer, AutoModel

import cfg

if __name__ == "__main__":
    args = cfg.parse_args()
