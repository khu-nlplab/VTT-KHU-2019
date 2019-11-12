# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

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


import tokenization
from modeling import SequenceClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, question, des, scene_des, local_scene_des, label=None):
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
        self.guid = guid
        self.question = question
        self.des = des
        self.scene_des = scene_des
        self.local_scene_des = local_scene_des
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, que_ids, des_ids, scene_ids, local_ids, label_id):

        self.que_ids = que_ids
        self.des_ids = des_ids
        self.scene_ids = scene_ids
        self.local_scene_ids = local_ids

        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if len(line) < 2:
                    pass
                else:
                    lines.append(line)

            return lines

class MemoryProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    def get_test_examples(self, data_dir):

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            tempstr = ''
            local_scene_des = []
            temp_list = list()
            guid = "%s-%s" % (set_type, i)
            que = tokenization.convert_to_unicode(line[0])
            des = tokenization.convert_to_unicode(line[4])

            for s_des in eval(line[5]):
                tempstr = tempstr + s_des

            if type(eval(line[5])) == type('string'):
                local_scene_des.append(tokenization.convert_to_unicode(eval(line[5])))
            elif type(eval(line[5])) == type(temp_list):
                for s_des in eval(line[5]):
                    local_scene_des.append(tokenization.convert_to_unicode(s_des))

            scene_des = tokenization.convert_to_unicode(tempstr)
            m_label = tokenization.convert_to_unicode(line[2])

            examples.append(
                InputExample(guid=guid, question=que, des=des,
                             scene_des=scene_des, local_scene_des=local_scene_des, label=m_label))
        return examples

class LogicalProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["2", "3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            tempstr = ''
            local_scene_des = []
            temp_list = list()
            guid = "%s-%s" % (set_type, i)
            que = tokenization.convert_to_unicode(line[0])
            des = tokenization.convert_to_unicode(line[4])

            for s_des in eval(line[5]):
                tempstr = tempstr + s_des

            if type(eval(line[5])) == type('string'):
                local_scene_des.append(tokenization.convert_to_unicode(eval(line[5])))
            elif type(eval(line[5])) == type(temp_list):
                for s_des in eval(line[5]):
                    local_scene_des.append(tokenization.convert_to_unicode(s_des))

            scene_des = tokenization.convert_to_unicode(tempstr)
            l_label = tokenization.convert_to_unicode(line[3])

            examples.append(
                InputExample(guid=guid, question=que, des=des,
                             scene_des=scene_des, local_scene_des=local_scene_des, label=l_label))
        return examples

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

        # local_scene_description
        local_ids = []
        for s_des in example.local_scene_des:
            tokens_L_scene_des = tokenizer.tokenize(s_des)
            if len(tokens_L_scene_des) > max_seq_length:
                tokens_L_scene_des = tokens_L_scene_des[0:(max_seq_length)]

            tokens = []

            for token in tokens_L_scene_des:
                tokens.append(token)

            Local_scene_ids = tokenizer.convert_tokens_to_ids(tokens)

            while len(Local_scene_ids) < max_seq_length:
                Local_scene_ids.append(0)

            assert len(Local_scene_ids) == max_seq_length

            local_ids.append(Local_scene_ids)

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                que_ids=que_ids,
                des_ids=des_ids,
                scene_ids=scene_ids,
                local_ids=local_ids,
                label_id=label_id))

    return features

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for develop")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=3000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--model_path',
                        type=str,
                        default='./model',
                        help='save model path')
    parser.add_argument('--load_model',
                        type=str,
                        default=None)
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300)
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=0.2)

    args = parser.parse_args()
    processors = {
        "memory": MemoryProcessor,
        "logic" : LogicalProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device((args.local_rank))
        device = "cuda"
        n_gpu = torch.cuda.device_count()
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
        torch.backends.cudnn.benchmark = True


    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

    vocab_dim = len(tokenization.load_vocab(args.vocab_file))

    model = SequenceClassification(vocab_dim, args.embedding_dim, args.dropout_prob, len(label_list), device)

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model,map_location='cpu'))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    global_step = 0

    if args.local_rank != -1:
        model = DDP(model)
        optimizer = FP16_Optimizer(optimizer)
        '''
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        '''
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        #train feature
        train_features = convert_to_ids(train_examples,label_list, args.max_seq_length, tokenizer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_q_ids = torch.tensor([f.que_ids for f in train_features], dtype=torch.long)
        all_d_ids = torch.tensor([f.des_ids for f in train_features], dtype=torch.long)
        all_sd_ids = torch.tensor([f.scene_ids for f in train_features], dtype=torch.long)
        #all_Ld_ids = torch.tensor([f.local_scene_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_q_ids, all_d_ids, all_sd_ids,  all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, num_workers=1, sampler=train_sampler, batch_size=args.train_batch_size)

        #developset feature
        dev_exmaples = processor.get_dev_examples(args.data_dir)
        dev_features = convert_to_ids(dev_exmaples, label_list, args.max_seq_length, tokenizer)

        all_dev_q_ids = torch.tensor([f.que_ids for f in dev_features], dtype=torch.long)
        all_dev_d_ids = torch.tensor([f.des_ids for f in dev_features], dtype=torch.long)
        all_dev_sd_ids = torch.tensor([f.scene_ids for f in dev_features], dtype=torch.long)
        #all_dev_Ld_ids = torch.tensor([f.local_scene_ids for f in dev_features], dtype=torch.long)
        all_dev_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

        dev_data = TensorDataset(all_dev_q_ids, all_dev_d_ids, all_dev_sd_ids,  all_dev_label_ids)
        if args.local_rank == -1:
            dev_sampler = RandomSampler(dev_data)
        else:
            dev_sampler = DistributedSampler(dev_data)

        dev_dataloader = DataLoader(dev_data, num_workers=1, sampler=dev_sampler, batch_size=args.eval_batch_size)

        model.train()
        losses = []
        dev_accuracy_list = []
        dev_losses = []
        for epoch in range(int(args.num_train_epochs)):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for (q_ids, d_ids, sd_ids, label_ids) in (train_dataloader):

                optimizer.zero_grad()

                q_ids = q_ids.to(device)
                d_ids = d_ids.to(device)
                sd_ids = sd_ids.to(device)
                #Ld_ids = Ld_ids.to(device)
                label_ids = label_ids.to(device)

                loss, _ = model.forward(q_ids, d_ids, sd_ids, label_ids)

                tr_loss += loss.item()
                nb_tr_examples += q_ids.size(0)
                nb_tr_steps += 1

                loss.backward()
                optimizer.step()

                global_step += 1

                if global_step%args.save_checkpoints_steps == 0:
                    if args.task_name == 'memory':
                        torch.save(model.state_dict(),
                               os.path.join(args.model_path, 'memory_model' + str(global_step) + '.bin'))
                    else:
                        torch.save(model.state_dict(),
                                   os.path.join(args.model_path,'logic_model'+str(global_step)+'.bin'))
            losses.append(tr_loss / nb_tr_steps)
            #develop dataset evaluation
            dev_accuracy, nb_dev_examples = 0, 0
            for q_ids, d_ids, sd_ids, label_ids in dev_dataloader:

                q_ids = q_ids.to(device)
                d_ids = d_ids.to(device)
                sd_ids = sd_ids.to(device)
                #Ld_ids = Ld_ids.to(device)
                label_ids = label_ids.to(device)

                dev_loss, logits = model.forward(q_ids, d_ids, sd_ids, label_ids)

                label_ids = label_ids.to('cpu').numpy()
                logits = logits.to('cpu').detach().numpy()

                tmp_dev_accuracy = accuracy(logits, label_ids)
                dev_accuracy += tmp_dev_accuracy

                nb_dev_examples += q_ids.size(0)

            print('-'*20)
            print("Epochs : {}".format(epoch+1))
            print("dev_accuracy : {}".format(dev_accuracy / nb_dev_examples))
            print("train Loss : {}".format(tr_loss/nb_tr_steps))
            print("validataion Loss : {}".format(dev_loss.item()))
            dev_losses.append(dev_loss.item())
            print('-' * 20)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_loss, = ax.plot([i for i in range(len(losses))], losses, label='train_loss')
    dev_loss, = ax.plot([i for i in range(len(dev_losses))], dev_losses, label='dev_loss')
    ax.legend()
    fig.savefig('[' + str(args.task_name) + '] loss.png')
    plt.close(fig)


    if args.do_eval:
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_to_ids(eval_examples, label_list, args.max_seq_length, tokenizer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_q_vectors = torch.tensor([f.que_ids for f in eval_features], dtype=torch.long)
        all_d_vectors = torch.tensor([f.des_ids for f in eval_features], dtype=torch.long)
        all_sd_vectors = torch.tensor([f.scene_ids for f in eval_features], dtype=torch.long)
        #all_Ld_vectors = torch.tensor([f.local_scene_ids for f in eval_features], dtype=torch.long)
        all_label_vectors = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_q_vectors, all_d_vectors, all_sd_vectors,  all_label_vectors)

        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, num_workers=1, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logit_label_list = []
        for step, (q_vec, d_vec, sd_vec, label_vec) in enumerate(tqdm(eval_dataloader, desc="Iteration")):

            q_vec = q_vec.to(device)
            d_vec = d_vec.to(device)
            sd_vec = sd_vec.to(device)
            #Ld_vec = Ld_vec.to(device)
            label_vec = label_vec.to(device)

            tmp_eval_loss, logits = model.forward(q_vec, d_vec, sd_vec, label_vec)

            label_ids = label_vec.to('cpu').numpy()
            logits = logits.to('cpu').detach().numpy()

            tmp_eval_accuracy = accuracy(logits, label_ids)

            output = np.argmax(logits, axis=1)

            list(output)
            list(label_ids)
            logit_label_list.append([output, label_ids])

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += q_vec.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps  # len(eval_dataloader)
        eval_accuracy = eval_accuracy / nb_eval_examples  # len(eval_dataloader)

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss / nb_tr_steps}  # 'loss': loss.item()}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open('[memory]align_epoch20_output', 'w') as f:
            logit_output_list = []
            Gold_output_list = []
            for labels in logit_label_list:
                for logit in labels[0]:
                    logit_output = convert_id_to_label(logit, label_list)
                    logit_output_list.append(logit_output)
                for Gold in labels[1]:
                    Gold_output = convert_id_to_label(Gold, label_list)
                    Gold_output_list.append(Gold_output)
            for logit, gold in zip(logit_output_list, Gold_output_list):
                f.write(str(logit) + '\t' + str(gold) + '\n')

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


def convert_id_to_label(label_, label_list):
    label_map = {}
    for (i, labels) in enumerate(label_list):
        label_map[i] = labels
    label = label_
    return label_map[label]


if __name__ == "__main__":
    main()
