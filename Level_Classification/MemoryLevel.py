from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random

import numpy as np
import torch
from torch .utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
#from prediction_model import BertConfig, BertForSequenceClassification
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

class InputExample(object):

    def __init__(self, guid, text_a, text_b, label = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MemoryProcessor(object):
    def get_labels(self):
        return ['1','2','3','4']

    def _create_examples(self, text_a, text_b, set_type):

        guid = '%s' % (set_type)
        text_a = tokenization.convert_to_unicode(text_a)
        text_b = tokenization.convert_to_unicode(text_b)
        example = InputExample(guid, text_a, text_b)
        return example
def convert_id_to_label(label_,label_list) :
    label_map = {}
    for (i, labels) in enumerate(label_list):
        label_map[i] = labels
    label = label_[0]
    return label_map[label]
def convert_examples_to_features(example, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i


    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = tokenizer.tokenize(example.text_b)

    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1]*len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if example.label == None:
        feature = InputFeatures(input_ids, input_mask, segment_ids)

    else:
        label_id = label_map[example.label]
        feature = InputFeatures(input_ids, input_mask, segment_ids, label_id)



    return feature

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length < max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def Memory_level_model(text_a, text_b):

    bert_config = BertConfig.from_json_file('uncased_L-12_H-768_A-12/bert_config.json')
    max_sequence_length = 128


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if max_sequence_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                max_sequence_length, bert_config.max_position_embeddings))
    processor = MemoryProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file='uncased_L-12_H-768_A-12/vocab.txt',do_lower_case=False)

    model = BertForSequenceClassification(bert_config,len(label_list))
    init_checkpoint = 'model/memory_model_500.bin'
    #Future save model Load code

    if init_checkpoint is not None:
        model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))

    model.to(device)

    eval_example = processor._create_examples(text_a, text_b,'input example')
    eval_feature = convert_examples_to_features(eval_example, label_list, max_sequence_length, tokenizer)

    input_ids = torch.tensor([eval_feature.input_ids],dtype=torch.long)
    input_mask = torch.tensor([eval_feature.input_mask], dtype=torch.long)
    segment_ids = torch.tensor([eval_feature.segment_ids], dtype=torch.long)
    if eval_feature.label_id == None :
        label_ids = None
    else :
        label_ids = torch.tensor(eval_feature.label_ids, dtype=torch.long)

    #eval_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    #eval_dataloader = DataLoader(eval_data)

    model.eval()

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    if label_ids == None :
        pass
    else :
        label_ids = label_ids.to(device)

    if label_ids == None:
        logits = model(input_ids, segment_ids, input_mask, label_ids)
    else :
        loss, logits = model(input_ids, segment_ids, input_mask, label_ids)
    logits = logits.detach().cpu().numpy()

    output = np.argmax(logits, axis=0)
    output = convert_id_to_label(output, label_list)
    return output





