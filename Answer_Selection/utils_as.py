""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import json
import tqdm
from io import open
from collections import Counter
from functools import reduce

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_q, text_d=None, text_sd=None, text_as=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_q: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_d: (Optional) The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            test_as: list of string. The untokenized text of the first sequence.

            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_q = text_q
        self.text_d = text_d
        self.text_sd = text_sd
        self.text_as = text_as
        self.label = label

class InputFeatures(object):
    """A single set of features of data"""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base of class for data converters for sequence classification data sets"""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_dev_example(self, question_text, clip_description_text, description_text, answer_candidates_list):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            samples = json.load(f)
            return samples

class AsProcessor(DataProcessor):
    def __init__(self, num_labels):
        self.num_labels = num_labels
    def get_train_examples(self, data_dir):
        data = self._read_json(os.path.join(data_dir, 'train.json'))

        examples = []


        for (i, line) in enumerate(data):
            index = i
            text_q = line['que']
            text_d = line['clip_desc']
            text_sd = line['scene_desc']
            text_as = [list(item.keys())[-1] for item in line['ans']]
            label = str([list(item.values())[-1] for item in line['ans']].index('true'))

            if text_d != "":
              examples.append(InputExample(index, text_q, text_d, text_sd, text_as, label))


        data = self._read_json(os.path.join(data_dir, 'valid.json'))

        for (i, line) in enumerate(data):
            index = i
            text_q = line['que']
            text_d = line['clip_desc']
            text_sd = line['scene_desc']
            text_as = [list(item.keys())[-1] for item in line['ans']]
            label = str([list(item.values())[-1] for item in line['ans']].index('true'))

            if text_d != "":
                examples.append(InputExample(index, text_q, text_d, text_sd, text_as, label))

            self.num_labels = len(label)


        return examples
    def get_dev_examples(self, data_dir):
        data = self._read_json(os.path.join(data_dir, 'test.json'))

        examples = []
        for (i, line) in enumerate(data):
            index = i
            text_q = line['que']
            text_d = line['clip_desc']
            text_sd = line['scene_desc']
            text_as = [list(item.keys())[-1] for item in line['ans']]
            label = str([list(item.values())[-1] for item in line['ans']].index('true'))
            if text_d != "":
              examples.append(InputExample(index, text_q, text_d, text_sd, text_as, label))

            self.num_labels = len(label)

        return examples

    def get_dev_example(self, question_text, clip_description_text, description_text, answer_candidates_list):
        examples = []
        text_q = question_text
        text_d = clip_description_text
        text_sd = description_text
        text_as = answer_candidates_list
        label = '0'
            
        self.num_labels = len(label)

        examples.append(InputExample(0, text_q, text_d, text_sd, text_as, label))
        return examples

    def get_labels(self):

        if self.num_labels:
            return [str(i) for i in range(self.num_labels)]

        return ['0', '1', '2', '3', '4']



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    q_tokens_length = []
    a_tokens_length = []
    cd_tokens_length = []
    sd_tokens_length = []
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc= "convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_list = []

        tokens_q = tokenizer.tokenize(example.text_q)
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # Answer Selection use only [SEP] so, Acount for [SEP[ with "-1"
        special_tokens_count = 4 if sep_token_extra else 3

        tokens_d = [""]
        if example.text_d:
            tokens_d =  tokenizer.tokenize(example.text_d)

        tokens_sd = [""]
        if example.text_sd:
            tokens_sd = tokenizer.tokenize(example.text_sd)

        tokens_as = []
        if example.text_as:
            for text_a in example.text_as:
                tokens_a = tokenizer.tokenize(text_a)
                tokens_as.append(tokens_a)
                tokens_list.append((tokens_d, tokens_sd, tokens_q ,tokens_a,))


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.



        dq_a_segment_ids_list = []
        dq_a_input_ids_list = []
        dq_a_input_mask_list = []

        sdq_a_segment_ids_list = []
        sdq_a_input_ids_list = []
        sdq_a_input_mask_list = []


        for tokens_d, tokens_sd, tokens_q, tokens_a in tokens_list:

            dq_tokens = tokens_d + tokens_q
            sdq_tokens = tokens_sd + tokens_q

            _truncate_seq_pair(dq_tokens, tokens_a, max_seq_length - special_tokens_count)
            _truncate_seq_pair(sdq_tokens, tokens_a, max_seq_length - special_tokens_count)

            dq_a_tokens = dq_tokens + [sep_token]
            sdq_a_tokens = sdq_tokens + [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                dq_a_tokens += [sep_token]
                sdq_a_tokens += [sep_token]

            dq_a_segment_ids = [sequence_a_segment_id] * len(dq_a_tokens)
            sdq_a_segment_ids = [sequence_a_segment_id] * len(sdq_a_tokens)

            dq_a_tokens += tokens_a + [sep_token]
            sdq_a_tokens += tokens_a + [sep_token]

            dq_a_segment_ids += [sequence_b_segment_id] * (len(tokens_a) + 1)
            sdq_a_segment_ids += [sequence_b_segment_id] * (len(tokens_a) + 1)

            if cls_token_at_end:
                dq_a_tokens = dq_a_tokens + [cls_token]
                sdq_a_tokens = sdq_a_tokens + [cls_token]

                dq_a_segment_ids = dq_a_segment_ids + [cls_token_segment_id]
                sdq_a_segment_ids = sdq_a_segment_ids + [cls_token_segment_id]

            else:
                dq_a_tokens = [cls_token] + dq_a_tokens
                sdq_a_tokens = [cls_token] + sdq_a_tokens

                dq_a_segment_ids = [cls_token_segment_id] + dq_a_segment_ids
                sdq_a_segment_ids = [cls_token_segment_id] + sdq_a_segment_ids

            dq_a_input_ids = tokenizer.convert_tokens_to_ids(dq_a_tokens)
            sdq_a_input_ids = tokenizer.convert_tokens_to_ids(sdq_a_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            dq_a_input_mask = [1 if mask_padding_with_zero else 0] * len(dq_a_input_ids)
            sdq_a_input_mask = [1 if mask_padding_with_zero else 0] * len(sdq_a_input_ids)

            # Zero-pad up to the sequence length.
            dq_a_padding_length = max_seq_length - len(dq_a_input_ids)
            sdq_a_padding_length = max_seq_length - len(sdq_a_input_ids)

            if pad_on_left:
                dq_a_input_ids = ([pad_token] * dq_a_padding_length) + dq_a_input_ids
                dq_a_input_mask = ([0 if mask_padding_with_zero else 1] * dq_a_padding_length) + dq_a_input_mask
                dq_a_segment_ids = ([pad_token_segment_id] * dq_a_padding_length) + dq_a_segment_ids

                sdq_a_input_ids = ([pad_token] * sdq_a_padding_length) + sdq_a_input_ids
                sdq_a_input_mask = ([0 if mask_padding_with_zero else 1] * sdq_a_padding_length) + sdq_a_input_mask
                sdq_a_segment_ids = ([pad_token_segment_id] * sdq_a_padding_length) + sdq_a_segment_ids

            else:
                dq_a_input_ids = dq_a_input_ids + ([pad_token] * dq_a_padding_length)
                dq_a_input_mask = dq_a_input_mask + ([0 if mask_padding_with_zero else 1] * dq_a_padding_length)
                dq_a_segment_ids = dq_a_segment_ids + ([pad_token_segment_id] * dq_a_padding_length)

                sdq_a_input_ids = sdq_a_input_ids + ([pad_token] * sdq_a_padding_length)
                sdq_a_input_mask = sdq_a_input_mask + ([0 if mask_padding_with_zero else 1] * sdq_a_padding_length)
                sdq_a_segment_ids = sdq_a_segment_ids + ([pad_token_segment_id] * sdq_a_padding_length)


            assert len(dq_a_input_ids) == max_seq_length
            assert len(dq_a_input_mask) == max_seq_length
            assert len(dq_a_segment_ids) == max_seq_length

            assert len(sdq_a_input_ids) == max_seq_length
            assert len(sdq_a_input_mask) == max_seq_length
            assert len(sdq_a_segment_ids) == max_seq_length


            dq_a_segment_ids_list.append(dq_a_segment_ids)
            dq_a_input_ids_list.append(dq_a_input_ids)
            dq_a_input_mask_list.append(dq_a_input_mask)

            sdq_a_segment_ids_list.append(sdq_a_segment_ids)
            sdq_a_input_ids_list.append(sdq_a_input_ids)
            sdq_a_input_mask_list.append(sdq_a_input_mask)

        q_tokens_length.append(len(tokens_q))
        a_tokens_length.append(sum([len(tokens_a) for tokens_a in tokens_as]) / 5)
        cd_tokens_length.append(len(tokens_d))
        sd_tokens_length.append(len(tokens_sd))

        segment_ids_list = dq_a_segment_ids_list + sdq_a_segment_ids_list
        input_ids_list = dq_a_input_ids_list + sdq_a_input_ids_list
        input_mask_list = dq_a_input_mask_list + sdq_a_input_mask_list

        if output_mode == "classification":
            label_id =  label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")  
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_q: %s" % " ".join(
                    [str(x) for x in tokens_q]))
            logger.info("tokens_d: %s" % " ".join(
                [str(x) for x in tokens_d]))
            logger.info("pair_input_ids: %s" % " ".join([str(x) for x in input_ids_list[0]]))
            logger.info("pair_input_mask: %s" % " ".join([str(x) for x in input_mask_list[0]]))
            logger.info("pair_segment_ids: %s" % " ".join([str(x) for x in segment_ids_list[0]]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids_list,
                              input_mask=input_mask_list,
                              segment_ids=segment_ids_list,
                              label_id=label_id,
                              ))
    print("q_avg_tokens_length : {}".format(sum(q_tokens_length)/len(examples)))
    print("a_avg_tokens_length : {}".format(sum(a_tokens_length) / len(examples)))
    print("cd_avg_tokens_length : {}".format(sum(cd_tokens_length) / len(examples)))
    print("sd_avg_tokens_length : {}".format(sum(sd_tokens_length) / len(examples)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):

    assert len(preds) == len(labels)
    if task_name == "as":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def write_predictions(examples, num_labels ,tokenizer, preds, labels, args):

    output_fail_pred_file = os.path.join(args.output_dir, "pred_fail_result.csv")
    output_true_pred_file = os.path.join(args.output_dir, "pred_true_result.csv")
    output_analysis_file = os.path.join(args.output_dir, "pred_result.txt")

    total_example_num = len(examples)
    wrong_example_num = 0
    total_num_cd_over_125= 0
    total_num_sd_over_125 = 0
    wrong_num_cd_over_125= 0
    wrong_num_sd_over_125 = 0
    total_cd_tokens_avg_length = 0
    total_sd_tokens_avg_length = 0
    wrong_cd_tokens_avg_length = 0
    wrong_sd_tokens_avg_length = 0

    q_token_length_list = []
    a_token_length_list = []
    cd_token_length_list = []
    sd_token_length_list = []


    fFile = open(output_fail_pred_file, "w")
    tFile = open(output_true_pred_file, "w")

    fWriter = csv.writer(fFile)
    tWriter = csv.writer(tFile)
    head = ["Question", "Clip_Description","Scene_Description"]
    head += ["Answer" + str(i)  for i in range(num_labels)]
    head += ["Prediction", "Label"]
    fWriter.writerow(head)
    tWriter.writerow(head)
    for example, pred, label in tqdm.tqdm(zip(examples, preds, labels)):

        assert str(example.label) == str(label)

        tokens_q = tokenizer.tokenize(example.text_q)
        tokens_cd = tokenizer.tokenize(example.text_d)
        tokens_sd = tokenizer.tokenize(example.text_sd)

        tokens_as = []
        if example.text_as:
            for text_a in example.text_as:
                tokens_a = tokenizer.tokenize(text_a)
                tokens_as.append(tokens_a)
        q_token_length_list.append(len(tokens_q))
        a_token_length_list.append(sum([len(tokens_a) for tokens_a in tokens_as]) / 5)
        cd_token_length_list.append(len(tokens_cd))
        sd_token_length_list.append(len(tokens_sd))

        cd_length_list = [len(tokens_a) + len(tokens_q) + len(tokens_cd) for tokens_a in tokens_as]
        sd_length_list = [len(tokens_a) + len(tokens_q) + len(tokens_sd) for tokens_a in tokens_as]
        cd_tokens_avg_length = sum(cd_length_list) / len(example.text_as)
        sd_tokens_avg_length = sum(sd_length_list) / len(example.text_as)
        num_cd_over_125 = 0
        num_sd_over_125 = 0

        for tokens_length in cd_length_list:
            if tokens_length > 125:
                num_cd_over_125 += 1
                break
        for tokens_length in sd_length_list:
            if tokens_length > 125:
                num_sd_over_125 += 1
                break

        total_cd_tokens_avg_length += cd_tokens_avg_length
        total_sd_tokens_avg_length += sd_tokens_avg_length
        total_num_cd_over_125 += num_cd_over_125
        total_num_sd_over_125 += num_sd_over_125

        if pred != label:
            wrong_example_num += 1
            wrong_cd_tokens_avg_length += cd_tokens_avg_length
            wrong_sd_tokens_avg_length += sd_tokens_avg_length
            wrong_num_cd_over_125 += num_cd_over_125
            wrong_num_sd_over_125 += num_sd_over_125

            row = [example.text_q, example.text_d, example.text_sd]
            row += example.text_as
            row += [str(pred), str(label)]
            fWriter.writerow(row)
        else:
            row = [example.text_q, example.text_d, example.text_sd]
            row += example.text_as
            row += [str(pred), str(label)]
            tWriter.writerow(row)

    with open(output_analysis_file, 'w') as fWriter:
        fWriter.write("total examples num : {} \n".format(total_example_num))
        fWriter.write("average Q tokens length : {} \n".format(sum(q_token_length_list) / total_example_num))
        fWriter.write("average A tokens length : {} \n".format(sum(a_token_length_list) / total_example_num))
        fWriter.write("average CD tokens length : {} \n".format(sum(cd_token_length_list) / total_example_num))
        fWriter.write("average SD tokens length : {} \n".format(sum(sd_token_length_list) / total_example_num))
        fWriter.write("total cdq_a tokens average length : {}\n".format(total_cd_tokens_avg_length / total_example_num))
        fWriter.write("total sdq_a tokens average length : {}\n".format(total_sd_tokens_avg_length / total_example_num))
        fWriter.write("number of over max cdq_a sequence length in all : {}\n".format(total_num_cd_over_125))
        fWriter.write("number of over max sdq_a sequence length in all : {}\n".format(total_num_sd_over_125))
        fWriter.write("wrong examples num : {} \n".format(wrong_example_num))
        fWriter.write("wrong cdq_a tokens average length : {}\n".format(wrong_cd_tokens_avg_length / wrong_example_num))
        fWriter.write("wrong sdq_a tokens average length : {}\n".format(wrong_sd_tokens_avg_length / wrong_example_num))
        fWriter.write("number of over max cdq_a sequence in wrong : {}\n".format(wrong_num_cd_over_125))
        fWriter.write("number of over max sdq_a sequence in wrong : {}\n".format(wrong_num_sd_over_125))
        


processors = {
    "as": AsProcessor,

}

output_modes = {
    "as": "classification",

}

GLUE_TASKS_NUM_LABELS = {
    "as": 5,

}
