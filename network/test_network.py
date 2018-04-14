#coding:utf-8
import tensorflow as tf
from .base_network import base_network


class test_network(base_network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.data = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=(3,))  # 图片信息，（高，宽，缩放比）
        self.keep_prob = tf.placeholder(tf.float32)
        self.setup()

    def setup(self):
        self.inputs = []
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        anchor_scales = 16
        _feat_stride = [16, ]
        # =========VGG16网络结构,与训练部分相同========
        (self.feed('data')
            .conv(3, 3, 64, 1, 1, name='conv1_1')
            .conv(3, 3, 64, 1, 1, name='conv1_2')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool1')
            .conv(3, 3, 128, 1, 1, name='conv2_1')
            .conv(3, 3, 128, 1, 1, name='conv2_2')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
            .conv(3, 3, 256, 1, 1, name='conv3_1')
            .conv(3, 3, 256, 1, 1, name='conv3_2')
            .conv(3, 3, 256, 1, 1, name='conv3_3')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool3')
            .conv(3, 3, 512, 1, 1, name='conv4_1')
            .conv(3, 3, 512, 1, 1, name='conv4_2')
            .conv(3, 3, 512, 1, 1, name='conv4_3')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool4')
            .conv(3, 3, 512, 1, 1, name='conv5_1')
            .conv(3, 3, 512, 1, 1, name='conv5_2')
            .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # ========= RPN ============
        # 注释详见train_network.py文件
        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        (self.feed('rpn_conv/3x3').bilstm(512, 128, 512, name='lstm_o'))  # 这里的512必须与最后一个卷积层的512匹配

        # 往两个方向走，一个用于给类别打分，一个用于盒子回归
        # 用于盒子回归的，输出是10个anchor，每个anchor有有2个回归，即y和高度,形状是[1, H, W, 20],
        # ===============“注意，网络输出的不是预测的盒子的四个坐标，而是y和高度的回归！！！！”========
        (self.feed('lstm_o').lstm_fc(512, self._cfg.COMMON.NUM_ANCHORS * 2, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512, self._cfg.COMMON.NUM_ANCHORS * 2, name='rpn_cls_score'))

        # 计算分数与回归
        # 'rpn_cls_score_reshape'里面装着softmax之前的得分，形状为(1, H, WxA, 2)
        # 'rpn_cls_score'里面则装着softmax以后的概率，形状为(1, H, WxA, 2)
        (self.feed('rpn_cls_score')
            .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
            .spatial_softmax(name='rpn_cls_prob'))

        # 输出softmax以后的概率，形状为(1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
            .spatial_reshape_layer(self._cfg.COMMON.NUM_ANCHORS * 2, name='rpn_cls_prob_reshape'))

        # 喂入的数据 分别是：
        """
        'rpn_cls_prob_reshape': softmax以后的概率值，形状为(1, H, W, Ax2)
        'rpn_bbox_pred': 回归，即y和高度,形状是[1, H, W, 20],
        'im_info': 图片信息，一个三维向量，包含高，宽，缩放比例
        """
        """
        该函数执行以后，添加进self.layers的有：
        'rpn_rois': (1 x H x W x A, 5) 第一列为正例的概率，后四列为映射回输入图片的，经过回归修正及nms以后的，预测的盒子坐标
        """
        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
            .proposal_layer(_feat_stride, name='rois'))


def get_test_network(cfg):
    return test_network(cfg)
