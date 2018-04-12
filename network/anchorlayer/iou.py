import numpy as np
from shapely.geometry import Polygon
"""
这里计算IOU的方法，是用交集的面积，除以（anchor的宽度乘以高的并）
这里，高的并的计算，是以上边和下边为依据的

"""


def get_y(x1, y1, x2, y2, x, min_val=True):
    if x1 == x2:
        if min_val:
            return min(y1, y2)
        else:
            return max(y1, y2)
    return (y2-y1)*(x-x1)/(x2-x1)+y1


def _next_ind(ind):
    assert 0 <= ind <= 3, "ind must be a valid index!"
    if ind <= 2:
        return ind + 1
    return 0


def _last_ind(ind):
    assert 0 <= ind <= 3, "ind must be a valid index!"
    if ind >= 1:
        return ind - 1
    return 3


def get_box_y(x1, y1, x2, y2, x3, y3, x4, y4, ctr_x):
    # x1, y1, x2, y2, x3, y3, x4, y4,构成了一个盒子的四个坐标，
    # 该函数要返回ctr_x处的竖直线与盒子交点的两个纵坐标
    X = np.array([x1, x2, x3, x4])
    Y = np.array([y1, y2, y3, y4])
    ints_sort = np.argsort(X, kind='mergesort')
    if X[ints_sort[0]] <= ctr_x <= X[ints_sort[1]]:
        cur_ind = ints_sort[0]
        last_ind = _last_ind(cur_ind)
        next_ind = _next_ind(cur_ind)
        x = X[cur_ind]
        y = Y[cur_ind]
        x_last = X[last_ind]
        y_last = Y[last_ind]
        x_next = X[next_ind]
        y_next = Y[next_ind]
        ymin = get_y(x, y, x_last, y_last, ctr_x, False)
        ymax = get_y(x, y, x_next, y_next, ctr_x, True)
    elif X[ints_sort[2]] <= ctr_x <= X[ints_sort[3]]:
        cur_ind = ints_sort[3]
        last_ind = _last_ind(cur_ind)
        next_ind = _next_ind(cur_ind)
        x = X[cur_ind]
        y = Y[cur_ind]
        x_last = X[last_ind]
        y_last = Y[last_ind]
        x_next = X[next_ind]
        y_next = Y[next_ind]
        ymin = get_y(x, y, x_last, y_last, ctr_x, True)
        ymax = get_y(x, y, x_next, y_next, ctr_x, False)
    else:
        ymin = get_y(x1, y1, x2, y2, ctr_x)
        ymax = get_y(x3, y3, x4, y4, ctr_x)
    return ymin, ymax


def bbox_overlaps(anchors, gt_boxes):
    assert anchors.shape[1] == 4, "in file {}, the number of anchor coordinates must be four".format(__file__)
    assert gt_boxes.shape[1] == 8, "in file {}, the number of gt coordinates must be eight".format(__file__)
    """
    Parameters
    ----------
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 8) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    N = anchors.shape[0]
    K = gt_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    gt_list = []

    for k in range(K):
        gt_coords = ((gt_boxes[k, 0], gt_boxes[k, 1]),
                     (gt_boxes[k, 2], gt_boxes[k, 3]),
                     (gt_boxes[k, 4], gt_boxes[k, 5]),
                     (gt_boxes[k, 6], gt_boxes[k, 7]),)
        gt_list.append(Polygon(shell=gt_coords))
    # ===========================================================================
    for n in range(N):  # 对anchor的遍历
        # 确保对于anchor而言，x2 >= x1
        assert anchors[n, 2] >= anchors[n, 0], "in file {}, x2 must be larger than x1 in anchor".format(__file__)
        anchor_width = anchors[n, 2] - anchors[n, 0] + 1  # anchor的宽度
        anchor_x = (anchors[n, 2] + anchors[n, 0])/2.0  # anchor的中心坐标
        an_coords = ((anchors[n, 0], anchors[n, 1]),
                     (anchors[n, 2], anchors[n, 1]),
                     (anchors[n, 2], anchors[n, 3]),
                     (anchors[n, 0], anchors[n, 3]),)
        an_box = Polygon(shell=an_coords)

        for k in range(K):  # 对gt_box的遍历
            if an_box.intersects(gt_list[k]):
                cross_area = gt_list[k].intersection(an_box).area
                # gt_y1 = get_y(gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2], gt_boxes[k, 3], anchor_x)
                # gt_y2 = get_y(gt_boxes[k, 4], gt_boxes[k, 5], gt_boxes[k, 6], gt_boxes[k, 7], anchor_x)

                gt_y1, gt_y2 = get_box_y(gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2], gt_boxes[k, 3],
                                         gt_boxes[k, 4], gt_boxes[k, 5], gt_boxes[k, 6], gt_boxes[k, 7], anchor_x)
                ymin = min([gt_y1, gt_y2, anchors[n, 1], anchors[n, 3]])
                ymax = max([gt_y1, gt_y2, anchors[n, 1], anchors[n, 3]])
                uh = ymax - ymin + 1

                overlaps[n, k] = cross_area / (uh*anchor_width)
    return overlaps


if __name__ == "__main__":
    from IPython import embed; embed()
