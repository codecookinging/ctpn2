import tensorflow as tf
from .base_network import base_network as bn


class train_network(bn):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 数据的输入入口,是一个形状为[批数，高，宽，通道数]的源图片，命名为“data”
        self.data = tf.placeholder(tf.float32, shape=[self._cfg.TRAIN.IMS_BATCH_SIZE, None, None, 3], name='data')
        # 图像信息，一个三维向量，包含图片张数，高，宽
        self.im_info = tf.placeholder(tf.float32, shape=(3,), name='im_info')
        # GT_boxes信息，N×8矩阵，每一行为一个gt_box，分别代表x1,y1,x2,y2,x3,y3,x4,y4,依次为左上，右上，右下，左下
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 8], name='gt_boxes')
        # dropout以后保留的概率
        self.keep_prob = tf.placeholder(tf.float32)
        self.setup()

    def setup(self):
        self.inputs = []
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        _feat_stride = [16, ]

        # padding本来是“VALID”，我把下面的padding全部改为了“SAME”， 以充分检测
        (self.feed('data')   # 把[批数，宽，高，通道]形式的源图像数据喂入inputs
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')   # k_h, k_w, c_o, s_h, s_w, name,
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
        # 在conv5_3中做滑动窗， 得到3×3×512的特征向量
        (self.feed('conv5_3')
             .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        # 将得到3×3×512的特征向量提取10个anchor并做双向LSTM中
        # 直接将划窗后的每个像素值送入到lstm中，由lstm提取出前后像素之间的联系
        (self.feed('rpn_conv/3x3').bilstm(512, 128, 512, name='lstm_o'))  # 这里的512必须与最后一个卷积层的512匹配
        # Bilstm的输出为[N, H, W, 512]形状
        # 经过lstm后，每个像素值将产生512维的向量，这些向量将被产生预测值

        # 往两个方向走，一个用于给类别打分，一个用于盒子回归
        # 用于盒子回归的，输出是10个anchor，每个anchor有有2个回归，即y和高度,形状是[1, H, W, 20],
        # ===============“注意，网络输出的不是预测的盒子的四个坐标，而是y和高度的回归！！！！”========
        (self.feed('lstm_o').lstm_fc(512, self._cfg.COMMON.NUM_ANCHORS * 2, name='rpn_bbox_pred'))

        # 用于anchor分类的，输出是[1, H, W, 20],
        (self.feed('lstm_o').lstm_fc(512, self._cfg.COMMON.NUM_ANCHORS * 2, name='rpn_cls_score'))

        """
        返回值如下
        rpn_labels是(1, FM的高，FM的宽，10),其中约150个值为0,表示正例; 150个值为1表示负例;其他的为-1,不用于训练
        rpn_bbox_targets 是(1, FM的高，FM的宽，20), 最后一个维度中，每2个表示一个anchor的回归 y,h
        这里是 GT与anchor之间的回归, 
        y的回归 = （GT的y-anchor的y）/anchor的高
        高的回归 = log(GT的高 / anchor的高)
        """
        # 将rpn_cls_score gt_boxes im_info 装入缓存
        # 经过这步之后，rpn_labels, rpn_bbox_targets将被得出，被存到self.layers['rpn-data']中，将在build loss阶段被取出加入到损失函数中
        # rpn_labels将被用来和预测值进行比较，具体为下一步softmax得出的rpn_cls_score_reshape
        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
             .anchor_target_layer(_feat_stride, name='rpn-data'))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name='rpn_cls_score_reshape')  # 把最后一个维度变成2,即(1,H,W,Ax2)->(1,H,WxA,2)
             .spatial_softmax(name='rpn_cls_prob'))  # 执行softmax，再转换为(1, H, WxA,2)


def get_train_network(cfg):
    return train_network(cfg)
