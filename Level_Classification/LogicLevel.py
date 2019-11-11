from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from matplotlib import pyplot as plt

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import FP16_Optimizer
from pickle_model import SequenceClassification

import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, question, des, scene_des, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.question = question
        self.des = des
        self.scene_des = scene_des
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, que_ids, des_ids, scene_ids, label_id):

        self.que_ids = que_ids
        self.des_ids = des_ids
        self.scene_ids = scene_ids
        self.label_id = label_id

class LogicalProcessor(object):
    def get_labels(self):
        """See base class."""
        return ["2", "3"]

    def _create_examples(self, qestion, clip_description, scene_description):
        """Creates examples for the training and dev sets."""
        examples = []
        examples.append(
                InputExample(question=qestion, des=clip_description,
                             scene_des=scene_description, label=None))
        return examples

def convert_id_to_label(label_,label_list) :
    label_map = {}
    for (i, labels) in enumerate(label_list):
        label_map[i] = labels
    label = label_[0]
    return label_map[label]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def convert_to_ids(examples, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):

        # question features
        tokens_que = tokenizer.tokenize(example.question)
        if len(tokens_que) > max_seq_length - 2:
            tokens_que = tokens_que[0:(max_seq_length)]
        tokens = []
        for token in tokens_que:
            tokens.append(token)

        que_ids = tokenizer.convert_tokens_to_ids(tokens)

        while len(que_ids) < max_seq_length:
            que_ids.append(0)

        assert len(que_ids) == max_seq_length

        # description features
        tokens_des = tokenizer.tokenize(example.des)
        if len(tokens_des) > max_seq_length:
            tokens_des = tokens_des[0:(max_seq_length)]
        tokens = []
        for token in tokens_des:
            tokens.append(token)
        des_ids = tokenizer.convert_tokens_to_ids(tokens)

        while len(des_ids) < max_seq_length:
            des_ids.append(0)
        assert len(des_ids) == max_seq_length


        # scene description
        tokens_scene_des = tokenizer.tokenize(example.scene_des)
        if len(tokens_scene_des) > max_seq_length:
            tokens_scene_des = tokens_scene_des[0:(max_seq_length)]

        tokens = []
        for token in tokens_scene_des:
            tokens.append(token)

        scene_ids = tokenizer.convert_tokens_to_ids(tokens)
        while len(scene_ids) < max_seq_length:
            scene_ids.append(0)

        assert len(scene_ids) == max_seq_length


        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                que_ids=que_ids,
                des_ids=des_ids,
                scene_ids=scene_ids,
                label_id=label_id))

    return features

def Logic_level_model(qestion, clip_description, scene_description):

    #environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_sequence_length = 128
    vocab_file = 'vocab.txt'
    embedding_dim = 200
    dropout_prob = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_dim = len(tokenization.load_vocab(vocab_file))

    processor = LogicalProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)

    label_list = processor.get_labels()


    model = SequenceClassification(vocab_dim, embedding_dim, dropout_prob, len(label_list), device)

    init_checkpoint = 'model/logic_model_500.bin'

    #Future save model Load code

    if init_checkpoint is not None:
        model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))

    model.to(device)

    eval_example = processor._create_examples(qestion, clip_description,scene_description)
    eval_feature = convert_to_ids(eval_example, label_list, max_sequence_length, tokenizer)

    que_ids = torch.tensor([f.que_ids for f in eval_feature],dtype=torch.long)
    des_ids = torch.tensor([f.des_ids for f in eval_feature], dtype=torch.long)
    scene_ids = torch.tensor([f.scene_ids for f in eval_feature], dtype=torch.long)

    if eval_feature.label_id == None :
        label_ids = None
    else :
        label_ids = torch.tensor([f.label_ids for f in eval_feature], dtype=torch.long)

    #eval_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    #eval_dataloader = DataLoader(eval_data)

    model.eval()

    que_ids = que_ids.to(device)
    des_ids = des_ids.to(device)
    scene_ids = scene_ids.to(device)
    if label_ids == None :
        pass
    else :
        label_ids = label_ids.to(device)
    if label_ids == None:
        logits = model(que_ids, des_ids, scene_ids, label_ids)
    else :
        loss, logits = model(que_ids, des_ids, scene_ids, label_ids)

    logits = logits.detach().cpu().numpy()

    output = np.argmax(logits, axis=0)
    output = convert_id_to_label(output, label_list)
    return output





