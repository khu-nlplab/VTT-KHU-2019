import tensorflow as tf
import numpy as np
from utils import utils


class Model(object):

    def __init__(self, token_emb_mat, glove_emb_mat, token_dict_size, char_dict_size, token_max_length, answer_max_nums, model_params, scope):
        self.scope = scope
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.token_emb_mat = token_emb_mat
        self.glove_emb_mat = glove_emb_mat
        self.model_params = model_params

        # batch * length
        self.que_token_seq = tf.placeholder(tf.int32, [None, None], name='que_tokens')
        self.des_token_seq = tf.placeholder(tf.int32, [None, None], name='des_tokens')

        # batch * answer_nums * length
        self.ans_token_seq = tf.placeholder(tf.int32, [None, None, None], name='ans_tokens')

        # batch * length * dimension
        self.que_char_seq = tf.placeholder(tf.int32, [None, None, None], name='que_chars')
        self.des_char_seq = tf.placeholder(tf.int32, [None, None, None], name='des_chars')

        # batch * answer_nums * length * char_legnth
        self.ans_char_seq = tf.placeholder(tf.int32, [None, None, None, None], name='ans_chars')


        self.token_dict_size = token_dict_size
        self.char_dict_size = char_dict_size
        self.token_max_length = token_max_length
        self.token_embedding_size = model_params.token_embedding_size
        self.char_embedding_size = model_params.char_embedding_size
        self.char_out_size = model_params.char_out_size
        self.out_char_dimensions = model_params.out_channel_dims
        self.filter_heights = model_params.filter_heights
        self.hidden_size = model_params.hidden_size
        self.finetune_emb = model_params.fine_tune

        self.dropout = model_params.drop_out

        self.que_token_mask = tf.cast(self.que_token_seq, tf.bool)
        self.des_token_mask = tf.cast(self.des_token_seq, tf.bool)
        self.ans_token_mask = tf.cast(self.ans_token_seq, tf.bool)

        self.que_token_len = tf.reduce_sum(tf.cast(self.que_token_mask, tf.int32), -1)
        self.des_token_len = tf.reduce_sum(tf.cast(self.des_token_mask, tf.int32), -1)
        self.ans_token_len = tf.reduce_sum(tf.cast(self.ans_token_mask, tf.int32), -1)

        self.que_char_mask = tf.cast(self.que_char_seq, tf.bool)
        self.des_char_mask = tf.cast(self.des_char_seq, tf.bool)
        self.ans_char_mask = tf.cast(self.ans_char_seq, tf.bool)

        self.que_char_len = tf.reduce_sum(tf.cast(self.que_char_mask, tf.int32), -1)
        self.des_char_len = tf.reduce_sum(tf.cast(self.des_char_mask, tf.int32), -1)
        self.ans_char_len = tf.reduce_sum(tf.cast(self.ans_char_mask, tf.int32), -1)

        self.answer_labels = tf.placeholder(tf.int32, [None], name='answer_label')
        self.output_class = answer_max_nums
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        self.wd = model_params.wd
        self.var_decay = model_params.var_decay
        self.decay = model_params.decay

        self.logits = None
        self.loss = None
        self.accuracy = None
        self.train_op = None
        self.summary = None
        self.optimizer = None
        self.ema = None
        self.var_ema = None

        self.update_tensor_add_ema_and_opt()

    def build_network(self):

        with tf.variable_scope('embedding'):

            que_emb = utils.token_and_char_emb(True, self.que_token_seq ,self.token_dict_size,
                                                    self.token_embedding_size, self.token_emb_mat,
                                                    self.glove_emb_mat,
                                                    True, self.que_char_seq, self.char_dict_size,
                                                    self.char_embedding_size, self.char_out_size,
                                                    self.out_char_dimensions, self.filter_heights,  False, None,
                                                    0., 1., True)
            des_emb = utils.token_and_char_emb(True, self.des_token_seq ,self.token_dict_size,
                                                    self.token_embedding_size, self.token_emb_mat,
                                                    self.glove_emb_mat,
                                                    True, self.des_char_seq, self.char_dict_size,
                                                    self.char_embedding_size, self.char_out_size,
                                               self.out_char_dimensions, self.filter_heights, False, None,
                                                    0., 1., True)

            # (batch*answer) * length * dimension
            answer_length = tf.shape(self.ans_token_seq)[1]

            ans_token_seq = tf.reshape(self.ans_token_seq,
                                            [-1,  tf.shape(self.ans_token_seq)[-1]])
            ans_char_seq = tf.reshape(self.ans_char_seq,
                                           [-1, tf.shape(self.ans_char_seq)[-2], tf.shape(self.ans_char_seq)[-1]])

            ans_emb = utils.token_and_char_emb(True, ans_token_seq ,self.token_dict_size,
                                                    self.token_embedding_size, self.token_emb_mat,
                                                    self.glove_emb_mat,
                                                    True, ans_char_seq, self.char_dict_size,
                                                    self.char_embedding_size, self.char_out_size,
                                                    self.out_char_dimensions, self.filter_heights, False, None,
                                                    0., 1., True)


        with tf.variable_scope('qd_interaction'):
            para, orth = utils.generate_para_orth(que_emb, des_emb, self.que_token_mask, self.des_token_mask,
                                                  scope='gene_para_orth', keep_prob=self.dropout, is_train=self.is_train,
                                                  wd=self.wd, activation='relu')


        with tf.variable_scope('qa_interaction'):

            q = tf.concat([para, orth], -1)
            a = utils.bn_dense_layer(ans_emb, 2 * self.hidden_size, True, 0., scope='ans_tanh',
                                          activation='tanh', enable_bn=False, wd=self.wd, keep_prob=self.dropout,
                                          is_train=self.is_train) * \
                     utils.bn_dense_layer(ans_emb, 2 * self.hidden_size, True, 0., scope='ans_sigmoid',
                                          activation='sigmoid', enable_bn=False, wd=self.wd, keep_prob=self.dropout,
                                          is_train=self.is_train)


            # batch * answer_nums * length * dimension
            q = tf.tile(tf.expand_dims(q,1), [1, answer_length, 1, 1])

            # (batch*answer_nums) * length * dimension
            q = tf.reshape(q, [-1, tf.shape(q)[-2], q.shape[-1]])

            self.que_token_mask = tf.tile(tf.expand_dims(self.que_token_mask,1), [1, answer_length,  1])
            self.que_token_mask = tf.reshape(self.que_token_mask,
                                             [-1,  tf.shape(self.que_token_mask)[-1]])

            self.ans_token_mask = tf.reshape(self.ans_token_mask, [-1, tf.shape(self.ans_token_mask)[-1]])


            q_inter, a_inter = utils.gene_qa_interaction(q, a, self.que_token_mask, self.ans_token_mask,
                                                         scope='qa_interaction', keep_prob=self.dropout, is_train=self.is_train,
                                                         wd=self.wd, activation='relu')

            q_vec = utils.multi_dimensional_attention(q_inter, self.que_token_mask, 'q_vec' , self.dropout,
                                                           self.is_train, 0., 'relu')
            a_vec = utils.multi_dimensional_attention(a_inter, self.ans_token_mask, 'a_vec' , self.dropout,
                                                           self.is_train, 0., 'relu')

        with tf.variable_scope('output'):

            # (batch*answer_nums) * dimension
            final_rep = tf.concat([q_vec, a_vec], -1)

            pre_logits = tf.nn.relu(utils.linear([final_rep], 300 , True, 0., scope='pre_logits',
                                                 squeeze=False, wd=self.wd, input_keep_prob=self.dropout,
                                                 is_train=self.is_train))

            logits = utils.linear([pre_logits], 1, True, 0., scope='logits',
                                squeeze=False, wd=self.wd, input_keep_prob=self.dropout,is_train=self.is_train)

            # batch * answer_nums
            logits = tf.reshape(logits,[-1, self.output_class])

            return logits

    def build_loss(self):
        with tf.variable_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

        # tf.logging.info("reg_vars num : %d " % len(reg_vars))
        # tf.logging.info("trainable_vars num : %d " % len(trainable_vars))

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.answer_labels,
            logits=self.logits,
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(tf.cast(tf.argmax(self.logits, -1), tf.int32),
                           self.answer_labels)

        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()

        if self.model_params.mode == 'train' or 'test':
            self.loss = self.build_loss()
            self.accuracy = self.build_accuracy()

            self.var_ema = tf.train.ExponentialMovingAverage(self.model_params.var_decay)
            self.build_var_ema()

            self.ema = tf.train.ExponentialMovingAverage(self.model_params.decay)
            self.build_ema()

            self.summary = tf.summary.merge_all()

            if self.model_params.optimizer.lower() == 'adadelta':
                assert self.model_params.learning_rate > 0.1 and self.model_params.learning_rate < 1.
                self.optimizer = tf.train.AdadeltaOptimizer(self.model_params.learning_rate)
            elif self.model_params.optimizer.lower() == 'adam':
                assert self.model_params.learning_rate < 0.1
                self.optimizer = tf.train.AdamOptimizer(self.model_params.learning_rate)
            else:
                raise AttributeError("no optimizer named as '%s' " % self.model_params.optimizer)

            self.train_op = self.optimizer.minimize(self.loss, self.global_step,
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_feed_dict(self, sample_batch, data_type='train'):
        que_token_seq_b = []
        que_char_seq_b = []

        des_token_seq_b = []
        des_char_seq_b = []

        ans_token_seq_b = []
        ans_char_seq_b = []


        qsl, dsl, asl = 0, 0, 0
        max_ans_num = 0
        for sample in sample_batch:
            qsl = max(qsl, len(sample['digitized_que_token']))
            dsl = max(dsl, len(sample['digitized_des_token']))

            asls = [len(per_sample) for per_sample in sample['digitized_ans_token']]
            asls.append(asl)
            asl = max(asls)

            max_ans_num = max(max_ans_num, len(sample['digitized_ans_token']))

        for sample in sample_batch:
            que_token = np.zeros([qsl], 'int32')
            que_char = np.zeros([qsl, self.token_max_length], 'int32')

            for idx_t, (token, char_seq_v) in enumerate(zip(sample['digitized_que_token'], sample['digitized_que_char'])):
                que_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.token_max_length:
                        que_char[idx_t, idx_c] = char

            des_token = np.zeros([dsl], 'int32')
            des_char = np.zeros([dsl, self.token_max_length], 'int32')

            for idx_t, (token, char_seq_v) in enumerate(
                    zip(sample['digitized_des_token'], sample['digitized_des_char'])):
                des_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.token_max_length:
                        des_char[idx_t, idx_c] = char

            ans_token = np.zeros([max_ans_num,asl],'int32')
            ans_char = np.zeros([ max_ans_num, asl, self.token_max_length], 'int32')

            for idx_a, (token_sample, char_sample) in enumerate(zip(sample['digitized_ans_token'], sample['digitized_ans_char'])):
                for idx_t, (token, char_seq_v) in enumerate(
                        zip(token_sample, char_sample)):
                    ans_token[idx_a,idx_t] = token
                    for idx_c, char in enumerate(char_seq_v):
                        if idx_c < self.token_max_length:
                            ans_char[idx_a,idx_t, idx_c] = char

            que_token_seq_b.append(que_token)
            que_char_seq_b.append(que_char)
            des_token_seq_b.append(des_token)
            des_char_seq_b.append(des_char)
            ans_token_seq_b.append(ans_token)
            ans_char_seq_b.append(ans_char)

        que_token_seq_b = np.stack(que_token_seq_b)
        que_char_seq_b = np.stack(que_char_seq_b)
        des_token_seq_b = np.stack(des_token_seq_b)
        des_char_seq_b = np.stack(des_char_seq_b)
        ans_token_seq_b = np.stack(ans_token_seq_b)
        ans_char_seq_b = np.stack(ans_char_seq_b)
        answer_label_b = None

        if data_type != 'prediction':
            answer_label_b = []
            for sample in sample_batch:
                answer_label_int = None
                for idx,answer in enumerate(sample['ans']):
                    if list(answer.values())[0] == 'true':
                        answer_label_int = idx
                        answer_label_b.append(answer_label_int)
                        answer_label_int = 0

                assert answer_label_int is not None

            answer_label_b = np.stack(answer_label_b).astype('int32')

            feed_dict= {self.que_token_seq: que_token_seq_b, self.que_char_seq: que_char_seq_b,
                    self.des_token_seq: des_token_seq_b, self.des_char_seq: des_char_seq_b,
                    self.ans_token_seq: ans_token_seq_b, self.ans_char_seq: ans_char_seq_b,
                    self.answer_labels: answer_label_b,
                    self.is_train: True if data_type == 'train' else False,
                    }

        else:
            feed_dict = {self.que_token_seq: que_token_seq_b, self.que_char_seq: que_char_seq_b,
                         self.des_token_seq: des_token_seq_b, self.des_char_seq: des_char_seq_b,
                         self.ans_token_seq: ans_token_seq_b, self.ans_char_seq: ans_char_seq_b,
                         self.is_train: True if data_type == 'train' else False,
                         }

        return feed_dict

    def step(self, sess, batch_samples, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')

        if get_summary:
            loss, summary, train_op = sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)

        else:
            loss,  train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None

        return loss, summary, train_op
