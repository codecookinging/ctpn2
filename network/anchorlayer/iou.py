import numpy as np


def bbox_overlaps(anchors, gt_boxes):
    assert anchors.shape[1] == 4
    assert gt_boxes.shape[1] == 4
    """
    Parameters
    ----------
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    N = anchors.shape[0]
    K = gt_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for n in range(N):  # 对anchor的遍历
        # anchor的面积
        # anchor_area = (anchors[n, 2] - anchors[n, 0] + 1)*(anchors[n, 3] - anchors[n, 1] + 1)
        anchor_width = (anchors[n, 2] - anchors[n, 0] + 1)
        for k in range(K):  # 对gt_box的遍历
            # x方向的交叉
            iw = min(anchors[n, 2], gt_boxes[k, 2]) - max(anchors[n, 0], gt_boxes[k, 0]) + 1
            if iw > 0:
                # 高度的交叉
                ih = min(anchors[n, 3], gt_boxes[k, 3]) - max(anchors[n, 1], gt_boxes[k, 1]) + 1
                if ih > 0:
                    uh = abs(gt_boxes[k, 3] - gt_boxes[k, 1]) + abs(anchors[n, 3] - anchors[n, 1])+2-ih
                    #
                    # gt_boxes_h = abs(gt_boxes[k, 3] - gt_boxes[k, 1])
                    na = float(iw * ih)  # 交集的面积
                    # denominator = iw * gt_boxes_h  # 用交集的宽度乘以gt的高度

                    # overlaps[n, k] = na/anchor_area  # 用交集的面积除以anchor的面积，作为衡量指标
                    # （交集的面积）除以（anchor的宽度和高度的并的乘积）作为IOU
                    overlaps[n, k] = na / (uh*anchor_width)
    return overlaps


if __name__ == "__main__":
    from IPython import embed; embed()
