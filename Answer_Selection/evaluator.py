import numpy as np
import tensorflow as tf

class Evaluator(object):
    def __init__(self, model, model_params, is_binary=True):
        self.model =  model
        self.global_step = model.global_step
        self.model_params = model_params
        self.is_binary = is_binary

        self.build_summary()
        self.writer = tf.summary.FileWriter(self.model_params.summary_dir)

    def get_evaluation(self, sess, dataset_obj, global_step=None):
        logits_list, loss_list, acc_list, prediction_list, label_list = [], [], [], [], []
        que_list, ans_list, des_list = [],[],[]

        for sample_batch,_, _, _ in dataset_obj.generate_batch_sample():
            feed_dict = self.model.get_feed_dict(sample_batch, dataset_obj.data_type)
            logits, loss, acc ,labels = sess.run(
                [self.model.logits, self.model.loss, self.model.accuracy, self.model.answer_labels], feed_dict)


            que_list += [sample['que'] for sample in sample_batch]
            if self.is_binary:
                ans_list += [sample['ans'] for sample in sample_batch]
            else:
                ans_list += [[list(answer.keys())[0] for answer in sample['ans']] for sample in sample_batch]



            des_list += [sample['description'] for sample in sample_batch]
            logits_list.append(np.argmax(logits, -1))
            loss_list.append(loss)
            acc_list.append(acc)

            prediction_list += np.argmax(logits, -1).astype('int32').tolist()
            label_list += np.array(labels).tolist()



        logits_array = np.concatenate(logits_list, 0)
        loss_value = np.mean(loss_list)
        acc_array = np.concatenate(acc_list, 0)
        acc_value = np.mean(acc_array)

        list_dict = {}
        list_dict['que_l'] = que_list
        list_dict['ans_l'] = ans_list
        list_dict['des_l'] = des_list
        list_dict['pre_l'] = prediction_list
        list_dict['lab_l'] = label_list

        if global_step is not None:
            if dataset_obj.data_type == 'train':
                summary_feed_dict = {
                    self.train_loss: loss_value,
                    self.train_accuracy: acc_value,

                }
                summary = sess.run(self.train_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            if dataset_obj.data_type == 'validation':
                summary_feed_dict = {
                    self.dev_loss: loss_value,
                    self.dev_accuracy: acc_value,

                }
                summary = sess.run(self.dev_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)


        return loss_value, acc_value, list_dict




    def build_summary(self):
        with tf.name_scope('train_summaries'):
            self.train_loss = tf.placeholder(tf.float32, [], 'train_loss')
            self.train_accuracy = tf.placeholder(tf.float32, [], 'train_accuracy')

            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss', self.train_loss))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy', self.train_accuracy))

            self.train_summaries = tf.summary.merge_all('train_summaries_collection')

        with tf.name_scope('dev_summaries'):
            self.dev_loss = tf.placeholder(tf.float32, [], 'dev_loss')
            self.dev_accuracy = tf.placeholder(tf.float32, [], 'dev_accuracy')

            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss', self.dev_loss))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy', self.dev_accuracy))
            self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')
