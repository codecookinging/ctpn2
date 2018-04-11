from .text_proposal_graph_builder import *


class TextProposalConnector(object):
    def __init__(self, cfg):
        self.graph_builder = TextProposalGraphBuilder(cfg)

    @staticmethod
    def fit_y(X, Y, x1, x2):
        assert len(X) != 0
        # 如果只包含一个点
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        # np.polyfit(X, Y, 1)用一阶多项式拟合离散点，返回拟合系数in decreasing powers
        p = np.poly1d(np.polyfit(X, Y, 1))
        # 返回在 x1,x2处的拟合直线的y坐标
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        :param text_proposals: 离散的文本片段
        :param scores: 对应的分数
        :param im_size: 缩放后的图片大小
        :return:返回一个N×9的矩阵，表示N个拼接以后的完整的文本框。
        每一行，前八个元素依次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
        """
        # 返回一个N阶方阵。若元素(i,j)为True，表示元素i与元素j相邻，并属于同一文本
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)

        # 将片段连接起来，返回一个列表。列表的每个元素是一个列表。元素列表包含了一个文本片段序列，该序列被预测为一个文本
        tp_groups = graphs_connected(graph)

        # 文本框的个数
        length = len(tp_groups)

        # 定义一个文本框矩阵，每行表示一个文本框，其中前四个元素表示坐标，最后一个表示文本框的分数
        text_lines = np.zeros((length, 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            # 获取一个文本片段序列
            text_line_boxes = text_proposals[list(tp_indices)]

            # 取出文本的首尾两端的x坐标
            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            # 返回第一个片段的宽度的一半
            offset = (text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            # 返回文本框上面一条边处的两个端点y值
            lt_y, rt_y = TextProposalConnector.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            # 返回文本框下面一条边处的两个端点y值
            lb_y, rb_y = TextProposalConnector.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # 文本框的分数被定义为所有文本片段的分数的平均值
            score = scores[list(tp_indices)].sum()/float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score

        # 将文本框裁剪，以不超过图片大小
        text_lines = clip_boxes(text_lines, im_size)

        text_recs = np.zeros((length, 9), np.float)
        index = 0
        for line in text_lines:
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            text_recs[index, 4] = xmin
            text_recs[index, 5] = ymax
            text_recs[index, 6] = xmax
            text_recs[index, 7] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes
