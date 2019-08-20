import tensorflow as tf
from absl import flags
from absl import app as absl_app
import os
import math

from utils import utils, utils_file
from model import Model as Model
from model_selector import Model as Model_Selector
import dataset
from model_params import Config

from graph_handler import GraphHandler
from evaluator import Evaluator
from perform_recoder import PerformRecoder

import preprocess_data

_TRAIN_TAG = "train"
_VALIDATION_TAG = "valid"
_TEST_TAG = "test"

_DATASET_TAGS = ['binary', 'multiple']

_JSON_TAG = 'json'

_DECODE_BATCH_SIZE = 16

def train(model_params):

    train_file = os.path.join(FLAGS.data_dir, _TRAIN_TAG + '_' + FLAGS.dataset_type + '.' + _JSON_TAG)
    validation_file = os.path.join(FLAGS.data_dir, _VALIDATION_TAG + '_' + FLAGS.dataset_type + '.' + _JSON_TAG)
    test_file = os.path.join(FLAGS.data_dir, _TEST_TAG + '_' + FLAGS.dataset_type + '.' + _JSON_TAG)

    filepaths = [train_file, validation_file, test_file]

    train_data, validation_data, test_data = utils_file.get_spilited_data(filepaths)

    dicts_file = os.path.join(model_params.dict_dir, 'dicts' + '_' + FLAGS.dataset_type + '.' + _JSON_TAG)

    if FLAGS.use_dicts:
        dicts = utils_file.read_file(dicts_file, _JSON_TAG)

    else:
        dicts = preprocess_data.build_dicts(train_data, model_params.glove_path, FLAGS.dataset_type)
        utils_file.write_file(dicts_file, dicts, _JSON_TAG)

    is_binary = True

    if FLAGS.dataset_type != 'binary':
        is_binary = False

    train_data = dataset.VTTExample(train_data, 'train', model_params, dicts, is_binary)
    valid_data= dataset.VTTExample(validation_data, 'validation', model_params, dicts, is_binary)
    # test_data = dataset.VTTExample(test_data, 'test', model_params, dicts, is_binary)

    emb_mat_token, emb_mat_glove = train_data.emb_mat_token, train_data.emb_mat_glove

    with tf.variable_scope('model') as scope:
        if FLAGS.dataset_type != 'multiple':
            model = Model(emb_mat_token, emb_mat_glove, len(train_data.dicts['token']), len(train_data.dicts['char']),
                    train_data.max_token_size, model_params=model_params, scope=scope.name)

        else:
            model = Model_Selector(emb_mat_token, emb_mat_glove, len(train_data.dicts['token']), len(train_data.dicts['char']),
                    train_data.max_token_size, train_data.max_ans_size, model_params, scope.name)

    model_params.load_model = FLAGS.is_load
    if FLAGS.load_path:
        model_params.load_path = FLAGS.load_path

    graphHandler = GraphHandler(model, model_params)
    evaluator = Evaluator(model, model_params, is_binary)
    perform_recoder = PerformRecoder(model_params)


    gpu_options = tf.GPUOptions()
    graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)

    graphHandler.initialize(sess)

    steps_per_epoch = int(math.ceil((train_data.sample_num / model_params.batch_size)))
    num_steps = FLAGS.num_steps or FLAGS.epochs * steps_per_epoch

    global_step = 0

    for sample_batch, batch_num, epoch, batch_idx in train_data.generate_batch_sample(num_steps):
        global_step = sess.run(model.global_step) + 1

        if_get_summary = global_step % (model_params.log_period) == 0
        loss, summary, train_op = model.step(sess, sample_batch, get_summary=if_get_summary)

        if global_step % 100 == 0:
            tf.logging.info('global_steps : %d' % global_step)
            tf.logging.info('loss : %.4f' % loss)

        if if_get_summary:
            graphHandler.add_summary(summary, global_step)

        if global_step % model_params.eval_period == 0 or epoch == FLAGS.epochs:

            train_loss, train_acc, train_dict = evaluator.get_evaluation(sess, train_data, global_step)
            tf.logging.info('train loss : %.4f accuracy %.4f' % (train_loss, train_acc))

            dev_loss, dev_acc, dev_dict = evaluator.get_evaluation(sess, valid_data, global_step)

            tf.logging.info('validation loss : %.4f accuracy %.4f' % (dev_loss, dev_acc))

            is_in_top, deleted_step = perform_recoder.update_top_list(global_step, dev_acc, sess)

            if train_acc - dev_acc > 0.02 :
                break

        if epoch == FLAGS.epochs:
            break

def test(model_params):

    model_params.batch_size = _DECODE_BATCH_SIZE

    model_params.load_model = FLAGS.is_load
    if FLAGS.load_path:
        model_params.load_path = FLAGS.load_path

    test_file = os.path.join(FLAGS.data_dir, _TEST_TAG + '_' + FLAGS.dataset_type + '.' + _JSON_TAG)
    test_data =utils_file.read_file(test_file, _JSON_TAG)

    dicts_path = os.path.join(model_params.dict_dir, 'dicts' + '_' + FLAGS.dataset_type + '.' + _JSON_TAG)

    dicts = utils_file.read_file(dicts_path, _JSON_TAG)

    assert dicts is not None

    is_binary = True

    if FLAGS.dataset_type != 'binary':
        is_binary = False

    test_data = dataset.VTTExample(test_data, 'test', model_params, dicts, is_binary)

    emb_mat_token, emb_mat_glove = test_data.emb_mat_token, test_data.emb_mat_glove

    with tf.variable_scope('model') as scope:
        if FLAGS.dataset_type != 'multiple':
            model = Model(emb_mat_token, emb_mat_glove, len(test_data.dicts['token']), len(test_data.dicts['char']),
                          test_data.max_token_size, model_params, scope=scope.name)

        else:
            model = Model_Selector(emb_mat_token, emb_mat_glove, len(test_data.dicts['token']),
                                   len(test_data.dicts['char']),
                                   test_data.max_token_size, test_data.max_ans_size, model_params, scope=scope.name)


    graphHandler = GraphHandler(model, model_params)
    evaluator = Evaluator(model, model_params, is_binary)

    gpu_options = tf.GPUOptions()
    graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)

    graphHandler.initialize(sess)

    test_loss, test_acc, test_dict = evaluator.get_evaluation(sess, test_data)

    tf.logging.info('test loss : %.4f accuracy %.4f' % (test_loss, test_acc))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    model_params  = Config(FLAGS.data_dir,FLAGS.model_dir, mode=FLAGS.mode )

    if FLAGS.mode == 'train':
        train(model_params)
    elif FLAGS.mode == 'test':
        test(model_params)
    else:
        raise RuntimeError('no running mode named as %s' % FLAGS.mode)


def define_flags():

    flags.DEFINE_string(
        name="data_dir", default=None, help="Directory path which has the dataset",
    )

    flags.DEFINE_integer(
        name='num_steps', default=None, help='max steps num'
    )

    flags.DEFINE_integer(
        name='epochs', default=None, help = 'epochs'
    )

    flags.DEFINE_string(
        name='mode', default=None, help='model mode(train, test)'
    )

    flags.DEFINE_bool(
        name='use_dicts', default=True,
        help="use preprocessed dictionary "
    )

    flags.DEFINE_string(
        name='glove_file', default='glove.6B.300d.txt',
        help='glove file name'
    )

    flags.DEFINE_bool(
        name='is_load', default=False,
        help='load model'
    )

    flags.DEFINE_string(
        name='load_path', default=None,
        help='load model path'
    )

    flags.DEFINE_string(
        name='model_dir', default=None,
        help='model path'
    )

    flags.DEFINE_string(
        name="dataset_type", default=None,
        help='The type of dataset is multiple(selecting one answer in some answers)'
             ' or binary(binary dataset)'
    )


    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required('mode')
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('dataset_type')

    @flags.validator('dataset_type',
                     message="dataset_type must be 'binary' or 'multiple'")
    def _check_dataset_type(provided_value):
        return True if str(provided_value) in _DATASET_TAGS else False

    @flags.multi_flags_validator(
        ['epochs', 'num_steps'],
        message="Both --epochs and --num_steps were set. Only one may be defined.")
    def _check_train_limits(flag_dict):
        return flag_dict['epochs'] is None or flag_dict['num_steps'] is None

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)
