import tensorflow as tf
import numpy as np
from .anchorlayer.anchor_target_tf import anchor_target_layer_py
from .anchorlayer.proposal_target_tf import proposal_layer as proposal_layer_py
DEFAULT_PADDING = "SAME"


# network中方法的专用装饰器
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs['name']
        # 取出输入数据
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # 执行操作，并返回输出数据
        layer_output = op(self, layer_input, *args, **kwargs)
        # 把输出结果加入到layers里面去，保存起来
        self.layers[name] = layer_output
        # 喂入临时缓存
        self.feed(layer_output)
        return self

    return layer_decorated


class base_network(object):
    """
    网络基类，核心属性有：
    inputs： 列表，用于存储临时数据
    layers： 字典，用于存储每一层的数据
    _cfg： 配置文件
    """
    def __init__(self, cfg):
        self.inputs = []  # 用于存储临时数据
        self.layers = dict()  # 用于存储每一层的数据
        self._cfg = cfg

    def feed(self, *args):
        assert len(args) != 0, "the data to feed cannot be empty!"
        self.inputs = []  # 每次喂入数据前，先将缓存清空
        for _layer in args:
            if isinstance(_layer, str):  # 从子类喂入
                data = self.layers[_layer]
                self.inputs.append(data)
            else:  # 从装饰器中喂入
                self.inputs.append(_layer)
        return self

    def load(self, data_path, session, ignore_missing=False):

        # data_dict是一个字典， 键为“conv5_1”, "conv3_2"等等
        # 而该字典的值又是字典，键为”biases"和"weights"
        data_dict = np.load(data_path, encoding='latin1').item()
        for key in data_dict.keys():

            # key 是“conv5_1”, "conv3_2"等等
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:
                            raise
    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,
             relu=True, padding=DEFAULT_PADDING, trainable=True):
        # input是上一层的数据，k_h, k_w为卷集核的高和宽， c_o为输出通道数，s_h, s_w为stride的高和宽
        c_i = input.get_shape()[-1]  # 返回输入通道数

        with tf.variable_scope(name) as scope:
            # 定义一个均值为零， 标准差为0.01的初始化器
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            # 定义常量0.0 的初始化器
            init_biases = tf.constant_initializer(0.0)
            # 相当于 tf.Variable() 这里cfg.TRAIN.WEIGHT_DECAY = 0.0005

            kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=init_weights,
                                     trainable=trainable, regularizer=base_network.l2_regularizer(self._cfg.TRAIN.WEIGHT_DECAY))

            if biased:
                biases = tf.get_variable(name='biases', shape=[c_o], initializer=init_biases, trainable=trainable)
                conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(conv, biases, name=scope.name)
            else:
                conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv

    @staticmethod
    def l2_regularizer(weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay, dtype=tensor.dtype.base_dtype, name='weight_decay')
                # tf.nn.l2_loss(t)的返回值是output = sum(t ** 2) / 2
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

        return regularizer

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def bilstm(self, input, d_i, d_h, d_o, name, trainable=True):
        # d_i是输入层维度512，d_h是隐层维度128，d_o是输出层维度512
        # 这里的input是由3×3的卷积核在512张通道图片中提取的特征， 是一个1×H×W×512的矩阵
        img = input
        with tf.variable_scope(name):
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])  # 第一次 d_i = 512

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            # 第一次d_h是128, 隐层的维度
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)

            # lstm_out是输出， last_state是隐层状态.lstm_out是一个元组，包含前向输出和后向输出(output_fw, output_bw)
            # output_fw和output_bw都是一个形状为[?, ?, 128]的张量
            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype=tf.float32)

            # 将output_fw和output_bw合并成一个[?, ?, 256]的张量
            lstm_out = tf.concat(lstm_out, axis=-1)

            # 每一行对应一个像素，即一个特征每个特征由256维向量表示，lstm_out是一个H×256的输出
            lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            # 初始化权重，权重是需要正则化的
            weights = tf.get_variable(name='weights', shape=[2 * d_h, d_o], initializer=init_weights,
                                      trainable=trainable,
                                      regularizer=base_network.l2_regularizer(self._cfg.TRAIN.WEIGHT_DECAY))
            # 偏执不需要正则化
            biases = tf.get_variable(name='biases', shape=[d_o], initializer=init_biases, trainable=trainable)

            # 全链接
            outputs = tf.nn.bias_add(tf.matmul(lstm_out, weights), biases)
            return tf.reshape(outputs, [N, H, W, d_o])

    @layer
    def lstm_fc(self, input, d_i, d_o, name, trainable=True):
        with tf.variable_scope(name):
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            # input的每一行代表一个像素， 第一次C=512
            input = tf.reshape(input, [N * H * W, C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)

            weights = tf.get_variable(name='weights', shape=[d_i, d_o], initializer=init_weights,
                                      trainable=trainable, regularizer=base_network.l2_regularizer(
                    self._cfg.TRAIN.WEIGHT_DECAY))
            # 偏置不需要正则化
            biases = tf.get_variable(name='biases', shape=[d_o], initializer=init_biases, trainable=trainable)

            out = tf.matmul(input, weights) + biases
            return tf.reshape(out, [N, H, W, int(d_o)])

    @layer
    def anchor_target_layer(self, input, _feat_stride, name):
        # input里面装着'rpn_cls_score', 'gt_boxes', 'im_info'
        # _feat_stride = [16,]
        # input的最后一个维度必须是3 ，即'rpn_cls_score', 'gt_boxes', 'im_info'
        assert len(input) == 3

        with tf.variable_scope(name):
            # 'rpn_cls_score', 'gt_boxes', 'im_info'

            """
            rpn_labels是(1, FM的高，FM的宽，10),其中约150个值为0,表示正例; 150个值为1表示负例;其他的为-1,不用于训练

            rpn_bbox_targets 是(1, FM的高，FM的宽，20), 最后一个维度中，每四个表示一个anchor的回归 y,h

            """
            # rpn_labels：(1, height, width, 10) height width为feature map对应的宽 高，一个像素只有一个标签
            # rpn_bbox_targets (1, height, width, 20) 标签为1的标签后回归目标

            rpn_labels, rpn_bbox_targets = tf.py_func(anchor_target_layer_py,
                                                      # input 分别对应 rpn_cls_score gt_boxes im_info
                                                      [input[0], input[1], input[2], _feat_stride],
                                                      [tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')

            # TODO 这里暂时只需要返回标签和anchor回归目标就可以了，后续会增加side refinement
            return rpn_labels, rpn_bbox_targets

    @layer
    def proposal_layer(self, input, _feat_stride,  name):
        # input 是一个包含三个元素的列表，包含一下三个元素值
        """
        'rpn_cls_prob_reshape': softmax以后的概率值，形状为(1, H, W, Ax2)
        'rpn_bbox_pred': 回归，即y和高度,形状是[1, H, W, 20],
        'im_info': 图片信息，一个三维向量，包含高，宽，缩放比例
        """
        # _feat_stride = [16,]
        assert len(input) == 3
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        with tf.variable_scope(name):
            # blob返回一個多行5列矩陣，第一列为分数，后四列为盒子坐标
            blob = tf.py_func(proposal_layer_py,
                                          [input[0], input[1], input[2], _feat_stride],
                                          [tf.float32])

            rpn_rois = tf.convert_to_tensor(tf.reshape(blob, [-1, 5]), name='rpn_rois')  # shape is (1 x H x W x A, 5)
            return rpn_rois

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input, [input_shape[0], input_shape[1], -1, int(d)], name=name)

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                              [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            return tf.nn.softmax(input, name=name)

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def build_loss(self):
        # 这一步输出的只是分数，没有softmax， 形状为(HxWxA, 2)
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])

        # self.get_output('rpn-data')[0]是形如(1, FM的高，FM的宽，10)的labels
        # 是真是的标签
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)

        # 取出标签为1 的label所在的索引，多行一列矩阵 shape=(?,1)
        fg_keep = tf.where(tf.equal(rpn_label, 1))

        # 取出标签为1 或者0的label所在的索引，多行一列矩阵
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        # 取出保留的标签所在行的分数
        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)  # shape (N, 2)

        # 取出保留的标签所在的标签
        rpn_label = tf.gather(rpn_label, rpn_keep)

        # 交叉熵损失
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

        # 預測的盒子回歸
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape [1, H, W, 20]

        # rpn_bbox_targets 是(1, FM的高，FM的宽，20) 最后一个维度中，每四个表示一个anchor的回归 y,h
        rpn_bbox_targets = self.get_output('rpn-data')[1]  # ================(1, FM的高，FM的宽，20)

        # 取出标签为1的盒子回归
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 2]), fg_keep)  # shape (N, 2)

        """这里是 GT与anchor之间的回归, 
        y的回归 = （GT的y-anchor的y）/anchor的高
        高的回归 = log(GT的高 / anchor的高)
        转换成两列的矩阵， 每一行为一个y和高度回归
        """
        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 2]), fg_keep)

        # 有内权重，是因为只需计算y值和高度的回归;有外权重，是因为只需计算正例的box回归
        rpn_loss_box_n = tf.reduce_sum(self.smooth_l1_dist((rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])

        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.cast(tf.shape(fg_keep)[0], tf.float32) + 1)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        model_loss = rpn_cross_entropy * 1000.0 + rpn_loss_box * 1000.0

        # 把正则化项取出来，以列表形式返回
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses)*80.0 + model_loss

        # 返回总损失（分类的交叉熵+盒子回归+正则化项）， 模型损失（分类的交叉熵+盒子回归）， 分类交叉熵， 盒子回归损失
        return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

