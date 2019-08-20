import tensorflow as tf
import numpy as np
import random
import math
import os

import tokenizer
from utils.utils_file import load_glove, write_file

PAD = '<pad>'
UNK = '<unk>'

PAD_ID = 0
UNK_ID = 1

class VTTExample(object):

    def __init__(self, data_list, data_type, model_params, dicts=None, is_binary=True):
        self.dataset_list = data_list
        self.data_type = data_type
        self.model_params = model_params
        self.sample_num = len(data_list)

        self.is_binary=is_binary

        assert data_type in ['train', 'validation', 'test', 'prediction'] and dicts is not None

        self.dicts = dicts
        self.max_lens = self.find_data_max_length(data_list)

        self.max_sentence_size = min(self.max_lens['sentence'], self.model_params.max_sentence_size)
        self.max_token_size = min(self.max_lens['token'], self.model_params.max_token_size)
        self.max_ans_size = min(self.max_lens['answer'],self.model_params.max_answer_num)

        self.digitized_data_list = self.digitize_dataset(data_list, self.dicts)


        self.emb_mat_token, self.emb_mat_glove = None, None

        if data_type != 'validation':
            self.emb_mat_token, self.emb_mat_glove = self.index2vector()

    def generate_batch_sample(self, max_step = None):
        if max_step is not None:
            def data_queue(data_list, batch_size):
                assert len(data_list) >= batch_size
                random.shuffle(data_list)
                data_pointer = 0
                step = 0
                batch_idx = 0
                epoch = 0

                while True:
                    if data_pointer + batch_size <= len(data_list):
                        yield data_list[data_pointer:data_pointer + batch_size], epoch , batch_idx
                        data_pointer += batch_size
                        batch_idx += 1
                        step += 1

                    elif data_pointer+ batch_size > len(data_list):
                        offset = data_pointer + batch_size - len(data_list)
                        batched_data_list = data_list[data_pointer:]
                        random.shuffle(data_list)
                        batched_data_list += data_list[:offset]
                        data_pointer = offset
                        epoch += 1
                        yield  batched_data_list, epoch ,batch_idx
                        step += 1
                        batch_idx += 1

                    if step >= max_step:
                        break

            batch_num = math.ceil(len(self.digitized_data_list) / self.model_params.batch_size)

            for sample_batch, epoch, batch_idx in data_queue(self.digitized_data_list, self.model_params.batch_size):
                yield sample_batch, batch_num, epoch, batch_idx
        else:
            # 단 1 epoch만 돈다.
            batch_num = math.ceil(len(self.digitized_data_list) / self.model_params.batch_size)
            batch_idx = 0
            sample_batch = []
            for sample in self.digitized_data_list:
                sample_batch.append(sample)
                if len(sample_batch) == self.model_params.batch_size:
                    yield  sample_batch, batch_num, 0, batch_idx
                    batch_idx += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield  sample_batch, batch_num, 0, batch_idx


    def index2vector(self):

        token2vector = load_glove(os.path.join(self.model_params.glove_dir, self.model_params.glove_file))

        lower_token2vector = {}
        for token, vec in token2vector.items():
            lower_token2vector[token.lower] = vec

        token2vector = lower_token2vector

        mat_token = np.random.normal(0, 1, size=(len(self.dicts['token']), self.model_params.token_embedding_size)).astype('float32')
        mat_glove = np.zeros((len(self.dicts['glove']), self.model_params.token_embedding_size), dtype="float32")

        for idx, token in enumerate(self.dicts['token']):
            try:
                mat_token[idx] = token2vector[token]
            except KeyError:
                pass

        mat_token[0] = np.zeros((self.model_params.token_embedding_size,), dtype="float32")


        for idx, token in enumerate(self.dicts['glove']):
            try:
                mat_glove[idx] = token2vector[token]
            except KeyError:
                mat_glove[idx] = np.zeros((self.model_params.token_embedding_size,), dtype="float32")

        tf.logging.info('mat_token_len : %d , mat_glove_len : %d' %(len(mat_token), len(mat_glove)))
        return mat_token, mat_glove

    def find_data_max_length(self, data_list):

        max_sentence = ''
        max_token = ''
        max_sentence_length = 0
        max_token_length = 0
        max_ans_num = 0

        for sample in data_list:
            question = sample['que']
            description = sample['description']
            sentence_collections = list()

            sentence_collections.append(question)
            sentence_collections.append(description)

            if self.is_binary:
                answer = sample['ans']
                sentence_collections.append(answer)
            else:
                answers = sample['ans']
                if self.data_type != 'prediction':
                    sentence_collections += [list(answer.keys())[0] for answer in answers]
                else:
                    sentence_collections += [answer for answer in answers]

                if len(answers) > max_ans_num:
                    max_ans_num = len(answers)

            for sentence in sentence_collections:
                tokens = tokenizer.sentence2token(sentence, lower=True)
                if len(tokens) > max_sentence_length:
                    max_sentence_length = len(tokens)
                    max_sentence = sentence
                    # print('sentence : %s \n max_sentence_length : %d' % (sentence,max_sentence_length))

                for token in tokens:
                    if len(token) > max_token_length:
                        max_token_length = len(token)
                        max_token = token
                        # print('word : %s \n max_sentence_length : %d' % (token,max_token_length))


        max_lens = {'sentence' : max_sentence_length, 'token' : max_token_length, 'answer' : max_ans_num}

        # tf.logging.info('-----finished finding the lengths-----')
        # print('sentence : %s \n max_sentence_length : %d' % (max_sentence, max_sentence_length))
        # print('word : %s \n max_token_length : %d' % (max_token, max_token_length))
        # print('max_answer_numbers : %d' %  (max_ans_num))

        return max_lens

    def digitize_dataset(self, data_list, dicts):

        token2index = dict([(token,idx) for idx, token in enumerate(list(dicts['token'] + dicts['glove']))])
        char2index = dict([(char,idx) for idx, char in enumerate(list(dicts['char']))])

        def digitize_token(token):
            try:
                return token2index[token]
            except KeyError:
                # UNK
                return 1

        def digitize_char(char):
            try:
                return char2index[char]
            except KeyError:
                # UNK
                return 1

        for sample in data_list:
            question = sample['que']
            description = sample['description']

            sample['digitized_que_token'] = [digitize_token(token) for token in tokenizer.sentence2token(question, True)]
            sample['digitized_que_char'] = [[digitize_char(char) for char in char_seq ] for char_seq in tokenizer.sentence2char(question, True)]
            sample['digitized_des_token'] = [digitize_token(token) for token in
                                             tokenizer.sentence2token(description, True)]
            sample['digitized_des_char'] = [[digitize_char(char) for char in char_seq] for char_seq in
                                            tokenizer.sentence2char(description, True)]


            if self.is_binary:
                answer = sample['ans']
                sample['digitized_ans_token'] = [digitize_token(token) for token in
                                                 tokenizer.sentence2token(answer, True)]

                sample['digitized_ans_char'] = [[digitize_char(char) for char in char_seq] for char_seq in
                                                tokenizer.sentence2char(answer, True)]
            else:
                answers = sample['ans']
                if self.data_type != 'prediction':
                    sample['digitized_ans_token'] = [[digitize_token(token)
                                                      for token in tokenizer.sentence2token(list(answer.keys())[0], True)]
                                                      for answer in answers]
                    sample['digitized_ans_char'] = [[[digitize_char(char) for char in char_seq] for char_seq in
                                                    tokenizer.sentence2char(list(answer.keys())[0], True)] for answer in answers]

                else:
                    sample['digitized_ans_token'] = [[digitize_token(token)
                                                      for token in
                                                      tokenizer.sentence2token(answer, True)]
                                                     for answer in answers]
                    sample['digitized_ans_char'] = [[[digitize_char(char) for char in char_seq] for char_seq in
                                                     tokenizer.sentence2char(answer, True)] for answer
                                                    in answers]

        return data_list


