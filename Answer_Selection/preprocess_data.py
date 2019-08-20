import json
import tensorflow as tf
import random
import math
import os
from absl import flags
from absl import app as absl_app

from utils.utils_file import read_file, write_file, make_dir, validate_path, find_file, get_spilited_data, load_glove

import tokenizer

DATA_FILE = "FriendsQA.json"

_TRAIN_TAG = "train"
_VALIDATION_TAG = "valid"
_TEST_TAG = "test"
_VOCAB_TAG = "vocab"

PAD = '<pad>'
UNK = '<unk>'

PAD_ID = 0
UNK_ID = 1

DATASET_TAGS = ['binary' , 'multiple']

_JSON_TAG = 'json'

random.seed(12345)

def split_answer(data, is_binary=False, valid_num_candidates = None):
    new_data = []

    for sample in data:
        num_candidates = len(sample['false_ans']) + 1

        if valid_num_candidates:
            assert  valid_num_candidates <= num_candidates

        new_sample = {}
        new_sample['que'] = sample['que']
        new_sample['description'] = sample['description']

        if is_binary:
            valid_num_candidates = 2


        new_sample['ans'] = []
        random.shuffle(sample['false_ans'])

        for i in range(valid_num_candidates):
            answer_label = {}

            if i == 0:
                answer_label[sample['true_ans']] = 'true'
            else:
                answer_label[sample['false_ans'][i-1]] = 'false'

            new_sample['ans'].append(answer_label)

        random.shuffle(new_sample['ans'])
        new_data.append(new_sample)

    return new_data

def split_dataset(raw_data, split_ratio, filepaths):

    data_len = len(raw_data)

    random.shuffle(raw_data)

    train_ratio = split_ratio[0]
    validation_ratio = split_ratio[1]
    test_ratio = split_ratio[2]

    train_len = math.ceil(data_len * train_ratio)
    validation_len = math.ceil(data_len * validation_ratio)
    test_len = data_len - train_len - validation_len

    train = raw_data[:train_len]
    validation = raw_data[train_len: train_len + validation_len]
    test = raw_data[-test_len:]

    if not validate_path(FLAGS.data_dir):
        make_dir(FLAGS.data_dir)

    write_file(filepaths[0], train, _JSON_TAG)
    write_file(filepaths[1], validation, _JSON_TAG)
    write_file(filepaths[2], test, _JSON_TAG)


def validate_dataset(filepaths):
    for fname in filepaths:
        if not validate_path(fname):
            return False
    return True

def make_vocab(inputs):

    token2index = {}
    tokens = set()
    index = 1
    for input in inputs:
        for que in input['que']:
            tokens.update([token.lower() for token in tokenizer.sentence2token(que)])
        for description in input['description']:
            tokens.update([token.lower() for token in tokenizer.sentence2token(description)])
        for false_ans in input['ans']:
            tokens.update([token.lower() for token in tokenizer.sentence2token(false_ans)])

    for token in tokens:
        token2index[token] = index
        index +=1

    return token2index

def analyze_data(dataset):

    result = {}

    number_of_questions = 0
    number_of_answers = 0
    number_of_descriptions = 0
    total_length_of_descriptions = 0
    total_length_of_questions = 0
    total_length_of_answers = 0

    for idx, data in enumerate(dataset):

        number_of_questions += 1
        total_length_of_questions += len(data['que'])

        number_of_answers += 1
        total_length_of_questions += len(data['ans'])

        number_of_descriptions += 1
        total_length_of_descriptions += len(data['description'])


    average_length_of_question = total_length_of_questions / number_of_questions
    average_length_of_answer = total_length_of_answers / number_of_answers
    average_length_of_description = total_length_of_descriptions / number_of_descriptions

    result["Number of questions"] = number_of_questions
    result["Number of answers"] = number_of_answers
    result["Number of descriptions"] = number_of_descriptions
    result["Average length of question"] = average_length_of_question
    result["Average length of answer"] = average_length_of_answer
    result["Average length of description"] = average_length_of_description

    return result

def build_dicts(data, glove_path, dataset_type=None):

    char_set = set()
    token_set = set()

    data_len = len(data)
    step = 0

    for sample in data:
        if step % 1000 == 0:
            tf.logging.info('buildindg dicts(%d/%d)' %(step , data_len))

        question = sample['que']
        description = sample['description']

        sentence_collections = list()
        sentence_collections.append(question)
        sentence_collections.append(description)

        if dataset_type != 'multiple':
            answer = sample['ans']
            sentence_collections.append(answer)

        else:
            answers = sample['ans']
            sentence_collections += [list(answer.keys())[0] for answer in answers]

        token_collections = list()
        char_collections = list()

        for sentence in sentence_collections:
            tokens = tokenizer.sentence2token(sentence, lower=True)
            token_collections += tokens
            for token in tokens:
                char_collections += list(token)

        step += 1

        char_set.update(char_collections)
        token_set.update(token_collections)

    char_set = list(char_set)
    token_set = list(token_set)

    glove_data = load_glove(glove_path)
    glove_tokens = glove_data.keys()
    glove_token_set = list(set([token for token in glove_tokens]))

    remove_token_list = []
    for token in token_set:
        if token in glove_token_set:
            remove_token_list.append(token)

    for token in remove_token_list:
        try:
            glove_token_set.remove(token)
        except:
            continue

    char_set.insert(PAD_ID, PAD)
    char_set.insert(UNK_ID, UNK)
    token_set.insert(PAD_ID, PAD)
    token_set.insert(UNK_ID, UNK)

    dicts = {'token': token_set, 'char': char_set, 'glove': glove_token_set}

    return dicts

def main(unused_argv):

    assert FLAGS.dataset_type in DATASET_TAGS

    dataset_type = FLAGS.dataset_type

    train_file = os.path.join(FLAGS.data_dir, _TRAIN_TAG + '_' + dataset_type +  '.' + _JSON_TAG)
    validation_file = os.path.join(FLAGS.data_dir, _VALIDATION_TAG + '_' + dataset_type + '.' + _JSON_TAG)
    test_file = os.path.join(FLAGS.data_dir, _TEST_TAG + '_' + dataset_type + '.' + _JSON_TAG)

    file_paths = [train_file, validation_file, test_file]

    # 데이터 원본 파일
    raw_data_file = FLAGS.data_file if not FLAGS.data_file else DATA_FILE
    raw_data_file_path = find_file(FLAGS.data_dir, raw_data_file)
    raw_data = read_file(raw_data_file_path, _JSON_TAG)

    is_binary = True

    if dataset_type == 'multiple':
        is_binary = False

    # 파일 존재 여부 검사
    if not validate_dataset(file_paths):

        # 파일 부재시 train, valid, test 나눔
        # 정답 선택에 따른 학습 데이터 구성
        preprocessed_ans_data = split_answer(raw_data, is_binary, FLAGS.num_ans)
        split_dataset(preprocessed_ans_data, FLAGS.split_ratio, file_paths)


    train, validation, test = get_spilited_data(file_paths)


    if FLAGS.analyze_data:
        train_result = analyze_data(train)
        validation_result = analyze_data(validation)
        test_result = analyze_data(test)

        total_result = list()

        total_result.append(train_result)
        total_result.append(validation_result)
        total_result.append(test_result)

        result_file = os.path.join(FLAGS.data_dir, "data_info")

        write_file(result_file, total_result)

    if FLAGS.build_dicts:
        glove_dir = os.path.join(FLAGS.data_dir, 'glove')
        glove_file = os.path.join(glove_dir, 'glove.6B.300d.txt')


        file_name = 'dicts' + '_' + FLAGS.dataset_type

        file_name = file_name + '.' + _JSON_TAG

        dicts = build_dicts(train, glove_file, dataset_type)

        dicts_path = os.path.join(FLAGS.data_dir,'dicts',file_name)

        write_file(dicts_path, dicts, _JSON_TAG)

def define_precess_data_flags():

    flags.DEFINE_string(
        name="data_dir", default='./data', help="Directory path which has the dataset",
    )
    flags.DEFINE_string(
        name="data_file",default=DATA_FILE, help="Raw data file name",
    )

    flags.DEFINE_multi_float(
        name="split_ratio", default = [0.8, 0.1, 0.1] , lower_bound=0.0, upper_bound=1.0,
        help="The three float values which divide the dataset"
    )

    flags.DEFINE_bool(
        name='make_vocab', default= False,
        help = "If True, it makes vocabulary"
    )

    flags.DEFINE_bool(
        name="analyze_data", default=False,
        help="If True, it analyze the dataset, else it doesn't."
    )

    flags.DEFINE_bool(
        name="build_dicts", default=True,
        help="If True, build word dictionary"
    )

    flags.DEFINE_string(
        name='dataset_type', default='binary',
        help='The type of dataset is multiple(selecting one answer in some answers)'
             ' or binary(binary dataset)'
    )

    flags.DEFINE_integer(
        name='num_ans', default=None,
        help="The number of answer candidates"
    )


    @flags.multi_flags_validator(
        ['dataset_type', 'num_ans'],
        message="If dataset_type is 'multiple', num_ans must not None.")
    def _check_train_limits(flag_dict):
        if flag_dict['dataset_type'] != 'binary' and flag_dict['num_ans'] == None:
            return False
        else:
            return True

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)
    define_precess_data_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)





