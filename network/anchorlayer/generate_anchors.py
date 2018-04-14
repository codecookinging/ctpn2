import numpy as np


def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = 7.5
    y_ctr = 7.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(cfg):
    # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    heights = np.logspace(start=0, stop=cfg.COMMON.NUM_ANCHORS, num=cfg.COMMON.NUM_ANCHORS,
                          base=cfg.COMMON.INCREASE_BASE, endpoint=False) * cfg.COMMON.MIN_ANCHOR_HEIGHT

    widths = 16
    sizes = []
    for h in heights:
        sizes.append((h, widths))
    return generate_basic_anchors(sizes)

