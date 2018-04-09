def get_polygonal_field(box):
    return round(triangle_field([box[0], box[1], box[2]]) + triangle_field([box[0], box[2], box[3]]))


def triangle_field(tri):
    group1 = (x1, x2, x3) = tri
    group2 = (x2, x3, x1)
    l1, l2, l3 = [line_len(p1, p2) for p1, p2 in list(zip(group1, group2))]
    s = (l1 + l2 + l3) / 2
    area = s * (s - l1) * (s - l2) * (s - l3)  # 海伦公式
    return pow(area, 0.5)


def line_len(p1, p2):
    return pow(pow(abs(p1[0] - p2[0]), 2) + pow(abs(p1[1] - p2[1]), 2), 0.5)


def inclination(p1, p2, f = True):
    '''f = true horizon'''
    if f:

        return round(abs(p2[0] - p1[0]) / abs(p2[1] - p1[1]), 1) if abs(p2[1] - p1[1]) != 0 else 'inf'
    else:
        return round(abs(p2[1] - p1[1]) / abs(p2[0] - p1[0]), 1) if abs(p2[0] - p1[0]) != 0 else 'inf'
