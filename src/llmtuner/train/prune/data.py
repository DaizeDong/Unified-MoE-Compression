# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import random

import numpy as np
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    traindata = load_dataset(
        'allenai/c4',  # 🔍 /mnt/petrelfs/dongdaize.d/quxioaye/data_for_t5/official_train
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
    )
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    valdata = load_dataset(
        'allenai/c4',  # 🔍 /mnt/petrelfs/dongdaize.d/quxioaye/data_for_t5/official_validation
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split='validation',
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_custom_dataset(dataset, nsamples, seed, seqlen):
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            # trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
            trainenc = {key: torch.tensor(value) for key, value in dataset[i].items()}  # 🔍
            if trainenc["input_ids"].shape[1] > seqlen:
                break
        i = random.randint(0, trainenc["input_ids"].shape[1] - seqlen - 1)
        j = i + seqlen
        input = trainenc["input_ids"][:, i:j]
        target = input.clone()
        target[:, :-1] = -100
        trainloader.append((input, target))

    # Prepare validation dataset
    # valenc = tokenizer(' '.join(dataset[:1100]['text']), return_tensors='pt')
    # valenc = valenc["input_ids"][:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)
    valenc = None
    return trainloader, valenc


"""这个文件没用了，可以删掉"""


# Function to select the appropriate loader based on dataset name
def get_loaders(name_or_dataset, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if isinstance(name_or_dataset, str):
        if 'wikitext2' in name_or_dataset:
            return get_wikitext2(nsamples, seed, seqlen, tokenizer)
        elif "c4" in name_or_dataset:
            return get_c4(nsamples, seed, seqlen, tokenizer)
        else:
            raise NotImplementedError
    else:
        return get_custom_dataset(name_or_dataset, nsamples, seed, seqlen)
