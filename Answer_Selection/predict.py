import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf
import json
import numpy as np

from model_selector import Model as Model_Selector
from model_params import Config
from dataset import VTTExample
from utils import utils_file
from graph_handler import GraphHandler

FLAGS = flags.FLAGS

_DECODE_BATCH_SIZE = 16
_JSON_TAG = 'json'

def _text2data(que, des, ans):
    data = {}
    data['que'] = que
    data['description'] = des
    data['ans'] = ans

    data_list = []
    data_list.append(data)

    return data_list

def _jsontext2data(text):

    try:
        data_list = json.loads(text)
    except Exception as ex:
        print("Error : ", ex)

    return data_list

def _file2data(file):

    return utils_file.read_file(file, _JSON_TAG)

def predict(data, model_params,
            print_all=True):

    with tf.variable_scope('model') as scope:
        model = Model_Selector(data.emb_mat_token, data.emb_mat_glove, len(data.dicts['token']), len(data.dicts['char']),
                           data.max_token_size, data.max_ans_size, model_params, scope.name)

    graphHandler = GraphHandler(model, model_params)

    gpu_options = tf.GPUOptions()
    graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)

    graphHandler.initialize(sess)

    logits_list, predictions_list = [], []
    que_list, ans_list, des_list = [], [], []
    output_list = []

    for sample_batch, _, _, _ in data.generate_batch_sample():

        feed_dict = model.get_feed_dict(sample_batch, 'prediction')
        logits = sess.run(model.logits, feed_dict=feed_dict)

        predictions = np.argmax(logits, -1)
        logits_list.append(logits)
        predictions_list.append(predictions)

        questions = [sample['que'] for sample in sample_batch]
        descriptions = [sample['description'] for sample in sample_batch]
        answers = [[answer for answer in sample['ans']] for sample in sample_batch]

        que_list.append(questions)
        des_list.append(descriptions)
        ans_list.append(answers)

        if print_all:
            for logit, prediction, answer_candidate in zip(logits, predictions, answers):
                tf.logging.info("Answer Confidence : {}\n".format(logit))
                tf.logging.info("Selected Answer : {}\n".format(answer_candidate[prediction]))
                output_list.append({'Answer_Confidence' : logit, 'Selected_Answer' : answer_candidate[prediction]})

    output_file = FLAGS.file_out
    if output_file is not None:
        if tf.gfile.IsDirectory(output_file):
            raise ValueError("File output is a directory")
        tf.logging.info("Writing to file {}".format(output_file))
        with tf.gfile.GFile(output_file, "w") as f:
            for output in output_list:
                f.write("Answer Confidence : {}\n".format(output['Answer_Confidence']))
                f.write("Selected Answer : {}\n".format(output['Selected_Answer']))



def main(unused_argv):

    model_params = Config(FLAGS.data_dir, FLAGS.model_dir)

    model_params.batch_size = _DECODE_BATCH_SIZE
    model_params.mode = 'prediction'

    model_params.load_model = FLAGS.is_load

    dicts = utils_file.read_file(model_params.dict_path, _JSON_TAG)

    if FLAGS.question_text is not None:
        data = _text2data(FLAGS.question_text, FLAGS.description_text, FLAGS.answer_candidates_list)
        encoded_data = VTTExample(data, 'prediction', model_params, dicts, False)

        predict(encoded_data, model_params)

    if FLAGS.json_text is not None:
        data = _jsontext2data(FLAGS.json_text)
        encoded_data = VTTExample(data, 'prediction', model_params, dicts, False)

        predict(encoded_data, model_params)

    if FLAGS.file is not None:
        data =_file2data(FLAGS.file)
        encoded_data = VTTExample(data, 'prediction', model_params, dicts, False)

        predict(encoded_data, model_params)


def define_predict_flags():
    flags.DEFINE_string(
        name="model_dir", default="./model",
        help="Directory containing answer selection checkpoints"
    )

    flags.DEFINE_string(
        name="data_dir", default=None,
        help="Directory containing token dictionary file. If prepreocess.py was used to encode training data,"
             "look in the data_dir to fine dictionary file"
    )
    flags.mark_flag_as_required('data_dir')

    flags.DEFINE_bool(
        name='is_load', default=False,
        help='Whether model use pretrained_model or not'
    )

    flags.DEFINE_string(
        name="question_text",default=None,
        help="Question text for answer selection"
    )

    flags.DEFINE_string(
        name="description_text", default=None,
        help="Description text for answer selection"
    )

    flags.DEFINE_multi_string(
        name="answer_candidates_list", default=None,
        help="Answer Candidates for answer selection"
    )

    flags.DEFINE_string(
        name='json_text', default=None,
        help="Input text in json format"
    )

    flags.DEFINE_string(
        name="file", default=None,
        help="File to predict, Output will be printed to console and if --file_output is provided"
             ",it will be saved to an output file."
    )

    flags.DEFINE_string(
        name='file_out', default=None,
        help='If --file_out is specified, save result to this file.'
    )

    @flags.multi_flags_validator(
        ['question_text', 'description_text', 'answer_candidates_list'],
        message="")
    def _check_text(flag_dict):
        if flag_dict['question_text'] is None and flag_dict['description_text'] is None and \
           flag_dict['answer_candidates_list'] is None :
            return True
        if flag_dict['question_text'] is not None and flag_dict['description_text'] is not None and \
           flag_dict['answer_candidates_list'] is not None:
            return True
        return False

    @flags.multi_flags_validator(
        ['question_text', 'json_text', 'file'],
        message="Nothing to predict")
    def _check_input(flag_dict):
        return not(flag_dict['question_text'] is None and flag_dict['json_text'] is None and flag_dict['file'] is None)

if __name__ == '__main__':
    define_predict_flags()
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)