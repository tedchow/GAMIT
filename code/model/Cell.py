import tensorflow as tf

class STRNNCell(tf.nn.rnn_cell.RNNCell):
    """A LSTM cell with multi-grained time and generalized graph convolutions instead of multiplications.

    Reference:
      Xingjian, SHI, et al. "Graph Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
      Bing Yu, et al. "Spatio-temporal graph convolutional neural network: A deep learning framework for traffic forecasting." IJCAI, 2018.

    """

    def __init__(self, isgraph, Lk, shape, filters, kernel, proj=None, \
                 forget_bias=1.0, activation=tf.tanh, normalize=True, \
                 peephole=True, data_format='channels_last', reuse=None, \
                 act_func='linear'):

        super(STRNNCell, self).__init__(_reuse=reuse)
        self.isgraph = isgraph
        self._kernel = kernel  # spatial,t kernel
        self.Ks = kernel[0]
        self.Kt = kernel[1]
        self.proj = proj
        self._filters = filters[0]
        self._gfilters= filters[1]
        self._forget_bias = forget_bias
        self._act_func = act_func
        self._activation = activation
        self._normalize = normalize
        self._peephole = peephole
        tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

        # print('filters', filters)
        if data_format == 'channels_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape([self._filters] + shape)
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
        return self._size


        
    def call(self, x, state):
        c, h = state
        # print('c:\t', c)
        # print('h:\t' ,h)
        # print('x shape:\t', x.get_shape())
        inputs = tf.concat([x, h], axis=self._feature_axis)
        inputs = tf.transpose(inputs, [0, 2, 1, 3])

        c_out = 4 * self._filters if self._filters > 1 else 4
        act_func = self._act_func

        fn = self._stgconv_fn
        K = self.Ks

        y = fn(inputs, K, c_out, act_func=act_func)
        y = tf.transpose(y, [0, 2, 1, 3])
        # print('graph conv y shape:\t', y.get_shape())  # (?, b_links, len_his2, 128)

        j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

        if self._peephole:
            i += tf.get_variable('W_ci', c.shape[1:]) * c
            f += tf.get_variable('W_cf', c.shape[1:]) * c

        if self._normalize:
            j = tf.contrib.layers.layer_norm(j)
            i = tf.contrib.layers.layer_norm(i)
            f = tf.contrib.layers.layer_norm(f)

        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
            o += tf.get_variable('W_co', c.shape[1:]) * c

        if self._normalize:
            o = tf.contrib.layers.layer_norm(o)
            c = tf.contrib.layers.layer_norm(c)

        o = tf.sigmoid(o)
        h = o * self._activation(c)
        # print ('h output:', h)
        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        if self.proj is not None:
            with tf.variable_scope("projection"):
                # h_t = _tconv_fn(self, h, self.config.len_his2, c_out, mask='VALID', scope=0):
                h = self._fully_con_layer(h, self._filters, self.proj)
                # print('output shape:\t', h.get_shape()) # h shape:         (?, 170, 4, 64)

        return h, state


    def gconv(self, x, theta, Ks, c_in, c_out):
        '''
        Spectral-based graph convolution function.
        :param x: tensor, [batch_size, n_route, c_in].
        :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
        :param Ks: int, kernel size of graph convolution.
        :param c_in: int, size of input channel.
        :param c_out: int, size of output channel.
        :return: tensor, [batch_size, n_route, c_out].
        '''
        # graph kernel: tensor, [n_route, Ks*n_route]
        kernel = tf.get_collection('graph_kernel')[0]
        n = tf.shape(kernel)[0]
        # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
        x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
        # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
        x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
        # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
        x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
        # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
        x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
        return x_gconv

    def spatio_gconv_layer(self, x, Ks, c_in, c_out, act_func='linear'):
        '''
        Spatial graph convolution layer.
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :param Ks: int, kernel size of spatial convolution.
        :param c_in: int, size of input channel.
        :param c_out: int, size of output channel.
        :return: tensor, [batch_size, time_step, n_route, c_out].
        '''
        _, T, n, _ = x.get_shape().as_list()
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> x:\t',x.get_shape())
        c_in = x.get_shape()[-1]
        if c_in > c_out:
            # bottleneck down-sampling
            w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
            x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
        elif c_in < c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
            x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
        else:
            x_input = x

        ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
        self.variable_summaries(ws, 'theta')
        bs = tf.get_variable('bs', [c_out], initializer=tf.zeros_initializer())

        # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
        x_gconv = self.gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
        # x_g -> [batch_size, time_step, n_route, c_out]
        x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> x_input:\t',x_input.get_shape())
        # print('x_gc:\t',x_gc.get_shape()) #[?, n_links, len_his, 32]
        # y =  x_gc[:, :, :, 0:c_out] + x_input
        # y = tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)
        if act_func == 'relu':
            return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)
        if act_func == 'linear':
            return x_gc[:, :, :, 0:c_out]
        if act_func == 'tanh':
            return tf.nn.tanh(x_gc[:, :, :, 0:c_out])
        if act_func == 'sigmoid':
            return tf.nn.sigmoid(x_gc[:, :, :, 0:c_out])
        else:
            raise ValueError('ERROR: activation function "%s" is not defined.' % act_func)
            # raise ValueError('ERROR')

    def variable_summaries(self, var, v_name):
        '''
        Attach summaries to a Tensor (for TensorBoard visualization).
        Ref: https://zhuanlan.zhihu.com/p/33178205
        :param var: tf.Variable().
        :param v_name: str, name of the variable.
        '''
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean_%s' % v_name, mean)

            with tf.name_scope('stddev_%s' % v_name):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev_%s' % v_name, stddev)

            tf.summary.scalar('max_%s' % v_name, tf.reduce_max(var))
            tf.summary.scalar('min_%s' % v_name, tf.reduce_min(var))

            tf.summary.histogram('histogram_%s' % v_name, var)


    def _gcn_fn(self, x, K, c_out, scope=0, act_func='linear'):
        """
        @param x: float tensor [?, len_his1, n_links, len_his2, c_in]
        @param c_out:
        """

        c_in = x.get_shape()[-1]
        y = self.spatio_gconv_layer(x, K, c_in, c_out, act_func=act_func)

        # W = tf.get_variable('graph_weights', [1, 1] + [c_in, c_out])
        # y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)

        #     weights = tf.get_variable(
        #         'weights', [input_size * num_matrices, output_size], dtype=dtype,
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
        return y

    def _conv_fn(self, x, K, c_out, scope=0, act_func='linear'):
        """  temporal conv
        @param x: float tensor [?, len_his1, n_links, len_his2, c_in]
        @param c_out:
        """
        c_in = x.get_shape()[-1]
        W = tf.get_variable('kernel', [self.Kt, self.Ks] + [c_in, c_out])
        y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)

        if not self._normalize:
            y += tf.get_variable('tbias_%s' % scope, [c_out], initializer=tf.zeros_initializer())

        return y

    def _tconv_fn(self, x, K, c_out, mask='SAME', scope=0, act_func='linear'):
        """  temporal conv
        @param x: float tensor [?, len_his2, n_links, c_in]
        @param c_out: VALID: [?, len_his2 + (K - 1), n_links,  c_in]
                        SAME: [?, len_his2, n_links,  c_in]
        """
        c_in = x.get_shape()[-1]
        Wt = tf.get_variable('convkernel_%s' % scope, [K, 1] + [c_in, c_out])
        y = tf.nn.convolution(x, Wt, mask, data_format=self._data_format)

        if not self._normalize:
            y += tf.get_variable('tbias_%s' % scope, [c_out], initializer=tf.zeros_initializer())
        if act_func == 'linear':
            return y
        if act_func == 'relu':
            return tf.nn.relu(y)
        if act_func == 'tanh':
            return tf.nn.tanh(y)
        if act_func == 'sigmoid':
            return tf.nn.sigmoid(y)

    def _stgconv_fn(self, x, K, c_out, scope=0, keep_prob=0, act_func='linear'):
        '''
        Spatio-temporal convolutional block, which contains two temporal gated convolution layers
        and one spatial graph convolution layer in the middle.
        :param x: tensor, batch_size, time_step, n_route, c_in].
        :param Ks: int, kernel size of spatial convolution.
        :param Kt: int, kernel size of temporal convolution.
        :param channels: list, channel configs of a single st_conv block.
        :param scope: str, variable scope.
        :param keep_prob: placeholder, prob of dropout.
        :param act_func: str, activation function.
        :return: tensor, [batch_size, time_step, n_route, c_out].
        '''
        c_t, c_oo = c_out, c_out

        with tf.variable_scope('stn_%s_in' % scope):
            x_s = self._tconv_fn(x, self.Kt, c_t, act_func=act_func)
            # print('x_s shape:\t', x_s.get_shape())
            x_t = self._gcn_fn(x_s, self.Ks, self._gfilters, act_func=act_func)
            # print('x_t shape:\t', x_t.get_shape())
        with tf.variable_scope('t_%s_out' % scope):
            x_o =  self._tconv_fn(x_t, self.Kt, c_oo)
        # x_ln = layer_norm(x_o, 'layer_norm_%s' % scope)
        # return tf.nn.dropout(x_ln, keep_prob=self.config.dr)
        return x_o


    def _fully_con_layer(self, x,  channel, c_out, scope=0):
        """
        @param n: int number of routes
        @param channel: n_route
        """
        w = tf.get_variable('fw_%s' % scope, [1, 1, channel, 1],initializer=tf.glorot_normal_initializer())
        # initializer=tf.truncated_normal_initializer()
        # w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
        # bs = tf.get_variable('bs', [c_out], initializer=tf.zeros_initializer())

        b = tf.get_variable('fb_%s' % scope, [1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        return tf.nn.convolution(x, w, 'SAME', data_format=self._data_format)  + b
