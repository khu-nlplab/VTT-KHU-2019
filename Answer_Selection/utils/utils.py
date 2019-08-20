import tensorflow as tf
import numpy as np
import json
from functools import reduce
from operator import mul

INF = 1e30


def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
                           extra_trainable=False, scope=None):

    with tf.variable_scope(scope or 'gene_emb_mat', reuse=tf.AUTO_REUSE):
        emb_mat_ept_and_unk = tf.constant(value=0, dtype=tf.float32, shape=[2, emb_len])
        if init_mat is None:
            emb_mat_other = tf.get_variable('emb_mat', [dict_size - 2, emb_len], tf.float32)
        else:
            tf.logging.info("init_mat : %s" % str(init_mat.shape))
            emb_mat_other = tf.get_variable('emb_mat', [dict_size - 2, emb_len], tf.float32,
                                            initializer=tf.constant_initializer(init_mat[2:], dtype=tf.float32,
                                                                                verify_shape=True))


        emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)

        if extra_mat is not None:
            if extra_trainable:
                extra_mat_var = tf.get_variable('extra_emb_mat', extra_mat.shape, tf.float32,
                                                initializer=tf.constant_initializer(extra_mat, dtype=tf.float32,
                                                                                    verify_shape=True))

                return tf.concat([emb_mat, extra_mat_var],0)
            else:
                tf.logging.info("extra_mat_con_size : %s" % str(extra_mat.shape))
                extra_mat_con = tf.get_variable(name='extra_emb_mat_var', shape=extra_mat.shape, dtype=tf.float32,
                                                initializer=tf.constant_initializer(extra_mat, dtype=tf.float32),
                                                trainable=False)
                return tf.concat([emb_mat, extra_mat_con], 0)

        else:
            return emb_mat

def scaled_tanh(x, scale=5.):
    return scale * tf.nn.tanh(1. / scale * x)

def generate_para_orth(que_emb, des_emb, que_mask, des_mask, scope=None,
                       keep_prob=1., is_train=None,  wd=0.,
                       activation='relu', name=None):
    batch_size, que_length, des_length, dimension_size = tf.shape(que_emb)[0], tf.shape(que_emb)[1], tf.shape(des_emb)[1], que_emb.get_shape()[2]

    with tf.variable_scope(scope or 'gene_para_orth'):
        head = tf.tile(tf.expand_dims(que_emb, 2), [1, 1, des_length, 1])
        tail = tf.tile(tf.expand_dims(des_emb, 1), [1, que_length, 1, 1])

        # batch * que_length * des_length * d
        parallel = head * tf.reduce_sum(head*tail, -1 ,True) / tf.reduce_sum(head*head, -1 ,True)
        orthogonal = tail - parallel

        parallel_dense = bn_dense_layer(parallel, dimension_size, bias=True, bias_start=0., scope='para_dense',
                                        activation=activation, enable_bn=False, wd=wd, keep_prob=keep_prob,
                                        is_train=is_train)
        orthogonal_dense = bn_dense_layer(orthogonal, dimension_size, bias=True, bias_start=0., scope='orth_dense',
                                        activation=activation, enable_bn=False, wd=wd, keep_prob=keep_prob,
                                        is_train=is_train)

        scaled_para = scaled_tanh(parallel_dense, 5.)
        scaled_orth = scaled_tanh(orthogonal_dense, 5.)

        # que_mask : batch * que_length
        # des_mask : batch * des_length
        # batch * que_length * des_length
        attention_mask = tf.cast(tf.expand_dims(que_mask ,-1),tf.int32) * tf.cast(tf.expand_dims(des_mask, 1), tf.int32)
        attention_mask = tf.cast(attention_mask, tf.bool)

        para_logits = exp_mask_for_high_rank(scaled_para, attention_mask)
        orth_logits = exp_mask_for_high_rank(scaled_orth, attention_mask)

        para_attention_score = tf.nn.softmax(para_logits, 2)
        para_attention_score = mask_for_high_rank(para_attention_score, attention_mask)
        para_attention_result = tf.reduce_sum(para_attention_score * tail , 2)



        orth_attention_score = tf.nn.softmax(orth_logits, 2)
        orth_attention_score = mask_for_high_rank(orth_attention_score, attention_mask)
        orth_attention_result = tf.reduce_sum(orth_attention_score * tail, 2)

        p_bias = tf.get_variable('p_bias', [dimension_size], tf.float32, tf.constant_initializer(0.))
        o_bias = tf.get_variable('o_bias', [dimension_size], tf.float32, tf.constant_initializer(0.))


        para_fusion_gate = tf.nn.sigmoid(
            linear(que_emb, dimension_size, True, 0., 'p_linear_fusion_input', False, wd, keep_prob, is_train) +
            linear(para_attention_result, dimension_size, False, 0., 'p_linear_fusion_attn', False, wd, keep_prob, is_train) +
            p_bias)

        para_output = para_fusion_gate * que_emb + (1 - para_fusion_gate) * para_attention_result
        para_output = mask_for_high_rank(para_output, que_mask)

        orth_fusion_gate = tf.nn.sigmoid(
            linear(que_emb, dimension_size, True, 0., 'o_linear_fusion_input', False, wd, keep_prob, is_train) +
            linear(orth_attention_result, dimension_size, False, 0., 'o_linear_fusion_attn', False, wd, keep_prob, is_train) +
            o_bias)
        orth_output = orth_fusion_gate * que_emb + (1 - orth_fusion_gate) * orth_attention_result
        orth_output = mask_for_high_rank(orth_output, que_mask)

        return para_output, orth_output


def gene_qa_interaction(q_rep_tensor, a_rep_tensor, q_rep_mask, a_rep_mask, scope=None,
                        keep_prob=1., is_train=None, wd=0.,
                        activation='relu', name=None):
    batch_size, que_length, ans_length, dimension_size = tf.shape(q_rep_tensor)[0], tf.shape(q_rep_tensor)[1], \
                                                         tf.shape(a_rep_tensor)[1], q_rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'gene_qa_inter'):
        head = tf.tile(tf.expand_dims(q_rep_tensor, 2), [1, 1, ans_length, 1])
        tail = tf.tile(tf.expand_dims(a_rep_tensor, 1), [1, que_length, 1, 1])

        #assert ans_length == a_rep_mask.get_shape()[1]

        attention_mask = tf.cast(tf.expand_dims(q_rep_mask, -1), tf.int32) * tf.cast(tf.expand_dims(a_rep_mask, 1), tf.int32)
        attention_mask = tf.cast(attention_mask, tf.bool)

        tf.print('attention_mask shape : %s' %str(attention_mask.get_shape().as_list()))
        qa_rep = tf.concat([head, tail], -1)
        interaction = bn_dense_layer(qa_rep, dimension_size, True, 0., 'interaction',
                                     activation, False, wd, keep_prob, is_train)

        scaled_inter = scaled_tanh(interaction, 5.)

        logits = exp_mask_for_high_rank(scaled_inter, attention_mask)

        inter_q_attention_score = tf.nn.softmax(logits, 2)
        inter_q_attention_score = mask_for_high_rank(inter_q_attention_score, attention_mask)
        inter_q_attention_result = tf.reduce_sum(inter_q_attention_score * tail, 2)

        inter_a_attention_score = tf.nn.softmax(logits, 1)
        inter_a_attention_score = mask_for_high_rank(inter_a_attention_score, attention_mask)
        inter_a_attention_result = tf.reduce_sum(inter_a_attention_score * head, 1)

        q_output = tf.concat([q_rep_tensor, inter_q_attention_result], -1)
        q_output = mask_for_high_rank(q_output, q_rep_mask)
        a_output = tf.concat([a_rep_tensor, inter_a_attention_result], -1)
        a_output = mask_for_high_rank(a_output, a_rep_mask)

        return q_output, a_output




def _linear(xs, output_size, bias, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs, -1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable("W", shape=[input_size, output_size], dtype=tf.float32)
        if bias:
            b = tf.get_variable('b', shape=[output_size], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + b
        else:
            out = tf.matmul(x, W)

        return out

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0., input_keep_prob=1., is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("'args' must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    # [-1, d]
    flat_args = [flatten(arg, 1) for arg in args]
    if input_keep_prob < 1.0:
        assert  is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]

    flat_out = _linear(flat_args, output_size, bias, bias_start, scope)
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep, dim_reduce_keep=None):
    dim_reduce_keep = dim_reduce_keep or keep

    # original shape
    ref_shape = ref.get_shape().as_list()
    # current shape
    tensor_shape = tensor.get_shape().as_list()
    # flatten dims list
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - dim_reduce_keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]

    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return  out

def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    tf.logging.info('scope_name : %s' % scope)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    tf.logging.info('variables length : %d' % len(variables))
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter

def highway_layer(arg,bias, bias_start= 0., scope=None, wd=0., input_keep_prob=1.0, is_train =None):
    with tf.variable_scope(scope or 'highway_layer'):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
                       is_train = is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
                      is_train = is_train)

        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1-gate) * arg
        return out


def highway_network(arg, num_layers , bias, bias_start=0., scope=None, wd=0., input_keep_prob=1., is_train=None):
    with tf.variable_scope(scope or 'highway_netwrok'):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start, scope='layer_{}'.format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur

def text_cnn(input, filter_sizes, filter_num, keep_prob=1.0):
    # batch * length * dimension * 1
    input_expand = tf.expand_dims(input, -1)
    sequence_length = tf.shape(input)[1]
    embedding_size = tf.shape(input)[2]
    pooled_output_list = []

    for filter_size in filter_sizes:
        with tf.variable_scope("conv{}".format(filter_size)):
            filter_shape = tf.convert_to_tensor([filter_size, embedding_size, 1, filter_num])

            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name='b')

            conv = tf.nn.conv2d(input_expand, W, [1, 1, 1, 1], 'VALID', name='conv')

            hidden = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')

            pooled = tf.nn.max_pool(hidden, ksize=[1, sequence_length - filter_size + 1 , 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

            pooled_output_list.append(pooled)

    total_filters_num = filter_num * len(filter_sizes)
    h_pool = tf.concat(pooled_output_list, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, total_filters_num])

    h_drop = tf.nn.dropout(h_pool_flat, keep_prob)

    return h_drop

def multi_dimensional_attention(rep_tensor, rep_mask, scope=None,
                   keep_prob=1., is_train=None, wd=0., activation='elu',
                   tensor_dict=None, name=None):
    batch_size, sequence_length, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):

        # batch * length * dimension
        map1 = bn_dense_layer(rep_tensor, ivec, bias=True, bias_start=0., scope='bn_dense_map1', activation=activation,
                              enable_bn=False, wd=wd, keep_prob=keep_prob, is_train=is_train)
        map2 = bn_dense_layer(map1, ivec, bias=True, bias_start=0., scope='bn_dense_map2', activation=activation,
                              enable_bn=False, wd=wd, keep_prob=keep_prob, is_train=is_train)


        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)

        # batch * vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)

        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output

def directional_attention_with_dense(rep_tensor, rep_mask, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tannh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    batch_size, sequence_length, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]

    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        sl_indices = tf.range(sequence_length, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction == None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sequence_length], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)

        # batch * length * length
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [batch_size, 1, 1])
        # batch * length * length
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sequence_length, 1])
        # batch * length * length
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)

        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        # batch * length * length * vec
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1,sequence_length, 1, 1])
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        with tf.variable_scope('attention'):
            f_bias = tf.get_variable_scope('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            # batch * length * vec
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')
            # batch * 1 * length * vec
            dependent_etd = tf.expand_dims(dependent, 1)
            # batch * length * vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head')
            # batch * length * 1 * vec
            head_etd = tf.expand_dims(head, 2)

            # batch * length * length * vec
            logits = scaled_tannh(dependent_etd + head_etd + f_bias, 5.0)

            logits_mask = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_mask, 2)
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train)+
                o_bias
            )
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)


        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate

        return output

def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    if activation == 'linear':
        activation_fun = tf.identity
    elif activation == 'relu':
        activation_fun = tf.nn.relu
    elif activation == 'elu':
        activation_fun = tf.nn.elu
    elif activation == 'selu':
        activation_fun = tf.nn.selu
    elif activation == 'sigmoid':
        activation_fun = tf.nn.sigmoid
    elif activation == 'tanh':
        activation_fun = tf.nn.tanh
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)

        if enable_bn:
            linear_map = tf.layers.batch_normalization(linear_map, center=True, training=is_train, name='bn')

        return activation_fun(linear_map)

def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        out = tf.nn.softmax(logits, -1)
        return out

def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')

def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask,-1)

    return tf.add(val, (1- tf.cast(val_mask, tf.float32)) * -INF,
                  name=name or 'exp_mask_for_high_rank')

def exp_mask(val, val_mask, name=None):
    if name is None:
        name="exp_mask"

    return tf.add(val, (1-tf.cast(val_mask, tf.float32)) * -INF, name=name)


def token_and_char_emb(if_token_emb=True, context_token=None, token_dict_size=None, token_embedding_length=None,
                       token_emb_mat=None, glove_emb_mat=None,
                       if_char_emb=True, context_char=None, char_dict_size=None, char_embedding_length=None,
                       char_out_size=None, filter_sizes=None, filter_heights=None, use_highway=True, highway_layer_num=None,
                       weight_decay=0., keep_prob=1., is_train=None, scope=None):

    with tf.variable_scope(scope or 'token_and_char_emb'):
        if if_token_emb:
            with tf.variable_scope('token_emb'):
                token_emb_mat = generate_embedding_mat(token_dict_size, token_embedding_length, init_mat=token_emb_mat,
                                                       extra_mat=glove_emb_mat, scope='gene_token_emb_mat')

                c_token_emb = tf.nn.embedding_lookup(token_emb_mat, context_token)

        if if_char_emb:
            with tf.variable_scope('char_emb'):
                char_emb_mat = generate_embedding_mat(char_dict_size, char_embedding_length, scope='gene_char_emb_mat')
                c_char_emb = tf.nn.embedding_lookup(char_emb_mat, context_char)

                assert sum(filter_sizes) == char_out_size and len(filter_sizes) == len(filter_heights)

                with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
                    c_char_emb = multi_conv1d(c_char_emb, filter_sizes, filter_heights, "VALID",
                                             is_train, keep_prob, scope="xx")

        if if_token_emb and if_char_emb:
            c_emb = tf.concat([c_token_emb, c_char_emb], -1)
        elif if_token_emb:
            c_emb = c_token_emb
        elif if_char_emb:
            c_emb = c_char_emb
        else:
            raise AttributeError('No embedding!')


    return c_emb

def dropout(input, keep_prob, is_train, noise_shape=None, seed=None, scope=None):
    with tf.name_scope(scope or "dropout"):
        assert is_train is not None

        if keep_prob < 1.0:
            d = tf.nn.dropout(input, keep_prob, noise_shape, seed)
            out = tf.cond(is_train, lambda :d , lambda : input)
            return out

        return input

def conv1d(input, filter_size, height, padding, is_train=None, keep_prob=1., scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = input.get_shape()[-1]
        filter = tf.get_variable('filter', shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable('bias', shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            input = dropout(input, keep_prob, is_train)

        xxc = tf.nn.conv2d(input, filter, strides, padding) + bias
        out = tf.reduce_max(tf.nn.relu(xxc) , 2)
        return out


def multi_conv1d(input, filter_sizes, heights, padding, is_train=None, keep_prob=1., scope=None):
    with tf.variable_scope(scope or 'multi_conv1d'):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes,heights):
            if filter_size == 0:
                continue

            out = conv1d(input, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope='conv1d_{}'.format(height))
            outs.append(out)
        concat_out = tf.concat(outs, 2)
        return concat_out