import numpy as np


def anchor_nms(height, width, proposals, scores, nms_thresh, fg_thresh):
    """
    对回归修正以后的anchors进行nms
    :param height 特征图的高
    :param width 特征图的宽
    :param proposals: shape是(N, 10*4) N表示像素的个数。每一行表示10个anchor,已经经过列回归修正
    :param scores: shape是(N, 10) 每一行表示10个anchor的得分
    :param nms_thresh: 阈值
    :param fg_thresh: 前景得分阈值
    :return: 两个列表，一个是nms以后的文字片段，一个是对应的分数
    """

    N = height*width  # 特征图的像素个数
    labels = np.empty(shape=(N, ), dtype=np.int8)
    labels.fill(-1)  # 标签初始化全部是-1
    new_proposal = np.empty((N, 4), proposals.dtype)
    new_scores = np.empty((N,), dtype=scores.dtype)

    index = 0
    # 针对每一行，找到每一行得分最大的anchor， 另外九个anchor的标签设置为-1
    # 得分最大的anchor，按照阈值来判为正例或者负例
    for line in scores:
        max_ind = np.argmax(line)
        # 取出得分最大的候选框
        new_proposal[index, :] = proposals[index, (4*max_ind):(4*max_ind+4)]
        # 取出得分最大的分数
        new_scores[index] = scores[index, max_ind]
        # 利用分数判断标签
        if scores[index, max_ind] >= fg_thresh:
            labels[index] = 1
        else:
            labels[index] = 0
        index += 1

    # 将标签，候选框，分数都变形为(height, width)状
    labels = labels.reshape((height, width))

    new_proposal = new_proposal.reshape((height, width, 4))
    new_scores = new_scores.reshape((height, width))
    nms_boxes = list()
    nms_scores = list()
    # 对每一列进行nms
    for i in range(width):
        if len(np.where(labels[:, i] == 1)[0]) == 0:
            continue
        # labels[:, i]的类型是数组

        box_list, score_list = \
            col_nms(list(new_proposal[:, i, :]), labels[:, i], list(new_scores[:, i]), nms_thresh)

        nms_boxes += box_list
        nms_scores += score_list

    return nms_boxes, nms_scores


def col_nms(boxes, labels, scores, nms_thresh):
    """
    对每列用非极大值抑制
    :param boxes: 一个长度为k的列表，列表的每个元素是一个候选框，包含四个坐标的np.array
    :param labels: 长度为k的对应的标签一维数组
    :param scores: 长度为k的对应的得分列表
    :param nms_thresh: nms阈值
    :return: 两个列表，第一个列表是进行非极大值抑制以后，留下来的候选框的坐标，第二个列表是对应的分数
    """
    fg_label = np.where(labels == 1)[0]

    new_boxes = [boxes[i] for i in fg_label]

    # 取出正例所在的分数
    new_score = [scores[i] for i in fg_label]
    # new_score = scores[fg_label]

    nms_boxes = list()  # 最终要返回的经过nms以后的候选框列表
    nms_score = list()

    while len(new_boxes) > 0:
        # max_ind = new_score.index(max(new_score))
        max_ind = int(np.argmax(np.array(new_score)))
        # 将得分最大的候选框纳入最终候选框，并从原来的列表中删除
        box = new_boxes.pop(max_ind)
        nms_boxes.append(box)
        nms_score.append(new_score.pop(max_ind))

        # 对其余的框框进行遍历
        if len(new_boxes) > 0:
            delete_index = []  # 容纳准备删除的元素的索引
            length = len(new_score)
            for i in range(length):
                # 将重叠度较大的非极大值，准备去掉
                if y_iou(new_boxes[i], box) > nms_thresh:
                    delete_index.append(i)

            scores_temp = [new_score[k] for k in range(length) if k not in delete_index]
            boxes_temp = [new_boxes[x] for x in range(length) if x not in delete_index]
            new_boxes = boxes_temp
            new_score = scores_temp

    assert len(nms_score) == len(nms_boxes)
    return list(nms_boxes), list(nms_score)


def y_iou(box1, box2):
    """
    计算y方向的iou
    :param box1: x1, y1, x2, y2
    :param box2:
    :return: y方向的iou
    """
    h1 = abs(box1[3]-box1[1])
    h2 = abs(box2[3]-box2[1])
    y0 = max(box1[1], box2[1])
    y1 = min(box1[3], box2[3])
    iou = (y1 - y0 + 1) / (h1 + h2 - (y1 - y0))
    # y方向的IOU
    return iou



