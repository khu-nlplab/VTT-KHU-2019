import os
from utils import utils_file


class Config(object):

    def __init__(self,
                 data_dir,
                 model_dir,
                 mode='train'):

        self.batch_size = 4
        self.token_embedding_size = 300
        self.char_embedding_size = 8
        self.hidden_size = 600

        self.data_dir = data_dir

        self.filter_heights = [3,4,5]
        self.out_channel_dims = [100, 100, 100]
        self.char_out_size = 300

        self.optimizer = 'adam'

        self.max_sentence_size = 256
        self.max_token_size = 16
        self.max_answer_num = 5

        self.fine_tune =False

        self.load_step = None
        self.load_model = False

        self.drop_out = 0.7
        self.learning_rate = 3e-5
        self.wd = 1e-6

        self.decay = 0.9
        self.var_decay = 0.999

        self.mode = mode
        self.log_period = 500
        self.eval_period = 1000

        self.model_ckpt_file='vttmodel.ckpt'
        self.dict_file='dicts_multiple.json'

        self.glove_dir = self.make_dir(os.path.join(self.data_dir, 'glove'))
        self.dict_dir = self.make_dir(os.path.join(self.data_dir, "dicts"))

        self.model_dir = self.make_dir(os.path.join(os.path.curdir, model_dir if model_dir else 'model'))

        self.result_dir = self.make_dir(os.path.join(self.model_dir, 'result'))

        self.log_dir = self.make_dir(os.path.join(self.model_dir, 'log'))
        self.summary_dir = self.make_dir(os.path.join(self.model_dir, 'summary'))
        self.ckpt_dir = self.make_dir(os.path.join(self.model_dir,'ckpt'))
        self.fail_dir = self.make_dir(os.path.join(self.model_dir,'fail'))
        self.answer_dir = self.make_dir(os.path.join(self.model_dir, 'answer'))

        self.dict_path = os.path.join(self.dict_dir, self.dict_file)
        self.ckpt_path = os.path.join(self.ckpt_dir, self.model_ckpt_file)
        self.load_path = None

        self.glove_file = 'glove.6B.300d.txt'
        self.glove_path = os.path.join(self.glove_dir, self.glove_file)

    def make_dir(self, path):

        utils_file.make_dir(path)

        return path