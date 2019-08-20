import tensorflow as tf

class GraphHandler(object):
    def __init__(self, model, model_params):
        self.model = model
        self.saver = tf.train.Saver()
        self.model_params =model_params
        self.writer = None

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        if self.model_params.load_model:
            self.restore(sess)
        if self.model_params.mode == 'train':
            self.writer = tf.summary.FileWriter(logdir=self.model_params.summary_dir, graph=tf.get_default_graph())


    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def save(self, sess, global_step = None):
        self.saver.save(sess, self.model_params.ckpt_path, global_step)


    def restore(self, sess):
        if self.model_params.load_step is None:
            if self.model_params.load_path is None :
                # latest_ckpt_path = tf.train.latest_checkpoint(self.model_params.ckpt_dir)

                latest_ckpt_path = ''

                ckpt_state = tf.train.get_checkpoint_state(self.model_params.ckpt_dir)
                ckpt_list = ckpt_state.all_model_checkpoint_paths

                latest_step = 0
                for ckpt in ckpt_list:
                    step = int((ckpt.split('-')[-1]).split('.')[0])
                    if step > latest_step:
                        latest_ckpt_path = ckpt
                        latest_step = step

            else:
                latest_ckpt_path = self.model_params.load_path

        else:
            latest_ckpt_path = self.model_params.ckpt_path + '-' + str(self.model_params.load_step)


        if latest_ckpt_path is not None:
            try:
                self.saver.restore(sess, latest_ckpt_path)

            except tf.errors.NotFoundError:
                if self.model_params.mode != 'train': raise FileNotFoundError('cannot restore model file')

        else:
            if self.model_params.mode != 'train': raise FileNotFoundError("cannot find model file")
