import tensorflow as tf
import numpy as np
from six.moves import zip

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from model.Cell  import STRNNCell
from model.basemodel import BaseModel
from utils.util import mkdir_file,  evaluation, z_inverse

def input_transformer_gcn(inputs, labels):   
    """ transform np array into tensor
    
    Parameters
    ----------
    inputs: float np array [?, len_his, n_links, len_his2, len_f]
    lables: float np array [?, len_pre, n_links, len_his2, len_f]

    Returns:
    ----------

    inputs: float tensor [?, len_pre, n_links, len_his2, len_f]
    lables: float tensor [?, len_pre, n_links, len_his2, len_f]
    """

    # inputs
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^input_transformer_gcn inputs:\t', inputs.get_shape().as_list())


    # labels
    if labels is None:
        pass
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        if 1:
            virt_shape = labels[0:-1].get_shape().as_list()
            virt_shape[1] = 1
            virtual_start = tf.zeros_like(labels[:,0:1,:,:,:], dtype=tf.float32, name="GO")
        else:
            virt_shape = labels.get_shape().as_list()
            virtual_start = inputs[:,-1:,:,:]
        print('labels shape:\t', labels.get_shape().as_list())

        # not useful when the loop function is employed
        lab_shape = labels.get_shape().as_list()
        labels = tf.concat([virtual_start, labels[:,0:-1,:,:]], axis=1)
        labels = tf.split(labels, lab_shape[1], 1)

    return inputs, labels

class STRNN(BaseModel):
    """docstring for STRNN"""

    def __init__(self, config, Lk, is_training=False):
        super(STRNN, self).__init__(config)
        self.config = config
        self.dropout_rate = self.config.dr
        self.is_training = is_training
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        # self.epoch_step = config.epoch_step
        # tf.Variable(0, trainable=False)
        self.Lk = Lk
        self.lambda_l2_reg=0.1

        preds = self.mod_fn()
        self.phs['preds'] = preds
        self.phs['loss'] = self.get_loss()
        tf.add_to_collection('loss', self.phs['loss'])
        self.phs['train_op'] = self.train_op()
        self.phs['summary'] = self.summary()

    def creat_encoder_strnn_cells(self, shape, filters, ks=3, kt=1):

        """
        Parameteres
        -----------
        shape: int list
        fileters: int list

        Returns:
        ----------
        MultiLyr_cell:
        """
        kernel = [ks, kt]
        with tf.variable_scope('Encoder'):
            stacked_cell = []

            for i in range(self.config.num_cell):
                with tf.variable_scope('STRNN_{}'.format(i)):
                    cell = STRNNCell(self.config.isgraph, self.Lk, shape, filters, kernel, act_func=self.config.act_func)
                    cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=1.0 - self.dropout_rate)
                    stacked_cell.append(cell)

        MultiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cell)

        return MultiLyr_cell

    def creat_decoder_strnn_cells(self, shape, filters, ks=3, kt=1):
        """
        Parameteres
        -----------
        shape: int list
        fileters: int list

        Returns:
        ----------
        MultiLyr_cell:
        """

        kernel = [ks, kt]
        with tf.variable_scope('Decoder'):
            stacked_cell = []

            for i in range(self.config.num_cell):
                with tf.variable_scope('STRNN_{}'.format(i)):
                    cell = STRNNCell(self.config.isgraph, self.Lk, shape, filters, kernel,
                                        proj=self.config.len_pre, act_func=self.config.act_func)
                    # cell = ConvLSTMCell(shape, filters, kernel)
                    cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=1.0 - self.dropout_rate)
                    stacked_cell.append(cell)

        MultiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cell)

        return MultiLyr_cell

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    def mod_fn(self):
        ###################### prepare some parameters ################################
        kernel = [3, 3]
        filters = self.config.filters
        batch_size = self.config.batch_size
        self.config.n_links = self.phs['labels'].get_shape().as_list()[2]
        self.config.len_f = self.phs['labels'].get_shape().as_list()[-1]
        shape = [self.config.n_links, self.config.len_his2]
        inputs_list, labels_list = input_transformer_gcn(self.phs['inputs'],
                                                         self.phs['labels'])

        with tf.variable_scope('STRNN_SEQ'):
            def _loop_function_in(prev, i):
                if self.is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if self.config.sampling:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(self.global_step, self.config.decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: tf.squeeze(labels_list[i], axis=1),
                                         lambda: prev)
                    else:
                        result = labels_list[i]
                    if len(result.get_shape()) > 4:
                        result = tf.squeeze(labels_list[i], axis=1)
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                return result

            ###################### encoder-decoder structure ##############################
            print('========================= Encoding!!!!!============================')
            with tf.variable_scope('Encoder_STRNN')  as scope:  # as BasicLSTMCell
                encoder_cells = self.creat_encoder_strnn_cells(shape, filters=filters, ks=self.config.Ks,
                                                                 kt=self.config.Kt)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cells,
                                                                   inputs=inputs_list,
                                                                   dtype=inputs_list.dtype)
                # _, enc_state = tf.nn.static_rnn(encoding_cells, inputs, dtype=tf.float32)

            print('========================= Decoding!!!!!============================')
            with tf.variable_scope('Decoder_STRNN')  as scope:  # as BasicLSTMCell
                decoder_cells = self.creat_decoder_strnn_cells(shape, filters=filters, ks=self.config.Ks,
                                                                 kt=self.config.Kt)
                outputs, final_state = self.rnn_decoder(labels_list, encoder_state, decoder_cells,
                                                        loop_function=_loop_function_in)

        preds = tf.concat(outputs, axis=1)
        preds = tf.reshape(preds,
                           [-1, self.config.len_pre, self.config.n_links, self.config.len_his2, self.config.len_f])
        # print('------------------------preds', preds.get_shape())

        return preds

    def rnn_decoder(self, decoder_inputs,
                    initial_state,
                    cell,
                    loop_function=None,
                    scope=None):
        """RNN decoder for the sequence-to-sequence model.
        Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
          in order to generate the i+1-st input, and decoder_inputs will be ignored,
          except for the first element ("GO" symbol). This can be used for decoding,
          but also for training to emulate http://arxiv.org/abs/1506.03099.
          Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x input_size].
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
        Returns:
        A tuple of the form (outputs, state), where:
          outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x output_size] containing generated outputs.
          state: The state of each cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
            (Note that in some cases, like basic RNN cell or LSTM cell, outputs and
             states can be the same. They are different for LSTM cells though.)
        """
        with tf.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            i = 0
            for inp in zip(decoder_inputs):
                # for i, inp in enumerate(decoder_inputs):
                inp = tf.squeeze(inp[0], axis=1)

                # print('---- inp ------', inp.get_shape())
                if loop_function is not None and prev is not None:
                    with  tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(inp, state)
                outputs.append(output)
                if loop_function is not None:
                    prev = output
                i += 1
        return outputs, state

    def loop_function_fcl(self, decoder_output):
        with tf.variable_scope('pre_Preds') as scope:  # as BasicLSTMCell
            # print('decoder_output shape 1:\t', decoder_output.get_shape()) # [?, n_links, 1, 32]
            s0, s1, s2, s3 = decoder_output.get_shape().as_list()
            decoder_output = tf.reshape(decoder_output, (-1, s1 * s2 * s3))
            # print('decoder_output 1:\t', decoder_output.get_shape()) #  (?, 1, 81, 1, 32)
            # print('self.phs[w_out] 1:\t', self.phs['w_out'].get_shape()) #  (?, 1, 81, 1, 32)

            output = tf.matmul(decoder_output, self.phs['w_out']) + self.phs['b_out']
            # print('*********************************pred shape 1:\t', decoder_output.get_shape())   # (?, len_pre, n_links, len_his2, len_f)
            # preds.append(decoder_output)
            return output

    def get_loss(self):
        n_steps_decoder = self.config.len_pre * self.config.len_his2
        batch_size = self.config.batch_size
        n_links = self.config.n_links
        len_his2 = self.config.len_his2

        preds = self.phs['preds']
        labels = self.phs['labels']

        # print('labels shape 1:\t', labels.get_shape())
        # print('preds shape 1:\t', preds.get_shape())

        labels = tf.reshape(tf.transpose(labels, [0, 1, 3, 2, 4]), [-1, n_steps_decoder, n_links])
        preds = tf.reshape(tf.transpose(preds, [0, 1, 3, 2, 4]), [-1, n_steps_decoder, n_links])

        labels = tf.split(labels, n_steps_decoder, 1)
        preds = tf.split(preds, n_steps_decoder, 1)
        # print('labels[0] shape 1:\t', labels[0].get_shape())

        # compute empirical loss
        empirical_loss = 0
        # Extra: we can also get separate error at each future time slot
        for _y, _Y in zip(preds, labels):
            __y = tf.reshape(_y, [-1, n_links])
            __Y = tf.reshape(_Y, [-1, n_links])
            empirical_loss += tf.reduce_mean(tf.pow(__y - __Y, 2))
        Lreg = self.lambda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.phs['empirical_loss'] = empirical_loss + Lreg

        # print(empirical_loss)
        print('empirical_loss:\t', empirical_loss)

        return empirical_loss

    def train_op(self):
        starter_learning_rate = self.config.lr
        learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                                                             self.global_step, self.config.decay_steps,
                                                             self.config.decay_rate, staircase=True)
        # lr = max(1e-6, learning_rate)
        # Passing global_step to minimize() will increment it at each step.
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.phs['loss'],
                                                                                           global_step=self.global_step)
        return optimizer

    def summary(self):
        tf.contrib.summary.scalar("loss", self.phs['loss'])
        for var in tf.trainable_variables():
            tf.contrib.summary.histogram(var.name, var)
        # print('summary done!!!')
        return tf.contrib.summary.all_summary_ops()
