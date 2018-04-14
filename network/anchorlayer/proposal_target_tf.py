from .generate_anchors import generate_anchors
from lib import load_config
import numpy as np
from .anchor_nms_pf import anchor_nms
cfg = load_config()


def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, _feat_stride=(16,)):

    """
    'rpn_cls_prob_reshape': softmax以后的概率值，形状为(1, H, W, Ax2)
    'rpn_bbox_pred': 回归，即y和高度,形状是[1, H, W, 20],
    'im_info': 图片信息，一个三维向量，包含高，宽，缩放比例
    cfg_key: 字符串， "TEST"
    _feat_stride = [16,]
     anchor_scales = [16,]
     cfg_key = 'TEST'

    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    """
    _anchors = generate_anchors(cfg)  # 生成基本的10个anchor
    _num_anchors = _anchors.shape[0]  # 10个anchor

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    nms_thresh = cfg.TEST.RPN_NMS_THRESH  # nms用参数，阈值是0.7
    min_size = cfg.TEST.RPN_MIN_SIZE  # 候选box的最小尺寸，目前是16，高宽均要大于16
    positive_thresh = cfg.TEST.TEXT_PROPOSALS_MIN_SCORE  # 大于这个分数阈值的判为正例
    # TODO 后期需要修改这个最小尺寸，改为8？

    height, width = rpn_cls_prob_reshape.shape[1:3]  # feature-map的高宽

    # 取出前景的得分，不去关心 背景的得分
    # (1, H, W, A) 这里取出的全部是前景的得分
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:, :, :, :, 1],
                        [1, height, width, _num_anchors])

    # 模型所输出的盒子回归
    bbox_deltas = rpn_bbox_pred  # 模型输出的pred是相对值，需要进一步处理成真实图像中的坐标

    # Enumerate all shifts
    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    A = _num_anchors
    K = shifts.shape[0]  # feature-map的像素个数
    anchors = _anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    bbox_deltas = bbox_deltas.reshape((-1, 2))  # (HxWxA, 2) 模型所输出的盒子回归值
    anchors = anchors.reshape((K * A, 4))  # 这里得到的anchor就是整张图像上的所有anchor
    proposals = bbox_transform_inv(anchors, bbox_deltas)  # 做逆变换，得到box在图像上的真实坐标

    proposals = proposals.reshape((K, 4*A))
    scores = scores.reshape((K, A))

    # 非极大值抑制，以列表形式输出进行列非极大值抑制后的文本片段以及相应的分数
    proposals, scores = anchor_nms(height, width, proposals, scores, nms_thresh, positive_thresh)

    proposals = np.array(proposals).reshape((-1, 4))
    scores = np.array(scores).reshape((-1, 1))

    # 对盒子进行裁剪，以保证不会超出图片边框
    proposals = clip_boxes(proposals, im_info[:2])  # 将所有的proposal修建一下，超出图像范围的将会被修剪掉

    # 移除那些proposal小于一定尺寸的proposal
    keep = _filter_boxes(proposals, min_size)
    proposals = proposals[keep, :]  # 保留剩下的proposal
    scores = scores[keep]
    #  score按得分的高低进行排序,返回脚标
    order = scores.ravel().argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]

    blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))
    # blob返回一個多行5列矩陣，第一行爲分數，後四行爲盒子坐標
    # bbox_deltas爲多行两列矩陣，每行爲一個回歸值
    return blob


def bbox_transform_inv(boxes, deltas):
    """

    :param boxes: shape是（H×W×10，4）每一行爲一個anchor的真實坐標
    :param deltas: （H×W×10，2）每行对应y个高度的回归
    :return:
    """

    # y的回归 = （GT的y - anchor的y） / anchor的高
    # 高的回归 = log(GT的高 / anchor的高)
    boxes = boxes.astype(deltas.dtype, copy=False)

    # widths = boxes[:, 2] - boxes[:, 0] + 1.0

    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    # ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dy = deltas[:, 0]
    dh = deltas[:, 1]

    pred_ctr_y = dy * heights + ctr_y
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(boxes.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0] = boxes[:, 0]
    # y1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h

    # x2
    pred_boxes[:, 2] = boxes[:, 2]
    # y2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    :param boxes: [N, 4]分别对应x1,y1,x2,y2
    :param im_shape: 二维向量，分别是图片的
    :return:
    """
    # x1 >= 0
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], im_shape[0] - 1), 0)
    return boxes


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where(hs >= min_size)[0]
    return keep
