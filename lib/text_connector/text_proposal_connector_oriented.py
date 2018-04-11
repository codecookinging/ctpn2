import numpy as np
from .text_proposal_graph_builder import *


class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self, cfg):
        self.graph_builder = TextProposalGraphBuilder(cfg)

    @ staticmethod
    def fit_y(X, Y, x1, x2):
        assert len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)


    # TODO 修改imsize
    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals: 许多离散的，映射到了最原始图片的文本片段
        scores：相应的得分
        im_size：图片的高，宽
        """

        # 首先还是建图，获取到文本行由哪几个小框构成
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        # 将片段连接起来，返回一个列表。列表的每个元素是一个列表。元素列表包含了一个文本片段序列，该序列被预测为一个文本
        tp_groups = graphs_connected(graph)

        length = len(tp_groups)

        # 0：文本行x坐标最小值；1：文本行y坐标的最小值
        # 2：文本行x坐标最大值；3：文本行y坐标的最大值
        # 4：整个文本行得分；   5：文本片段中心坐标拟合直线的斜率
        # 6：直线截距           7：文本行的高度
        text_lines = np.zeros((length, 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            # 获取一个文本片段序列
            text_line_boxes = text_proposals[list(tp_indices)]
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
            
            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0=np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1=np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            # 返回第一个片段的宽度的一半
            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y=TextProposalConnector.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y=TextProposalConnector.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            score=scores[list(tp_indices)].sum()/float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4]=score  # 文本行得分
            text_lines[index, 5]=z1[0]  # 根据中心点拟合的直线的斜率k
            text_lines[index, 6]=z1[1]  # 根据中心点拟合的直线的截距b
            height = np.mean( (text_line_boxes[:,3]-text_line_boxes[:,1]) )  # 小框平均高度

            # TODO 加2.5是什么意思？？
            text_lines[index, 7]= height + 2.5

        text_recs = np.zeros((length, 9), np.float)
        index = 0
        # 0：文本行x坐标最小值；1：文本行y坐标的最小值
        # 2：文本行x坐标最大值；3：文本行y坐标的最大值
        # 4：整个文本行得分；   5：文本片段中心坐标拟合直线的斜率
        # 6：直线截距           7：文本行的高度
        for line in text_lines:
            # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b1 = line[6] - line[7] / 2
            b2 = line[6] + line[7] / 2

            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上

            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上

            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下

            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下

            disX = x2 - x1  # 文本行x坐标最大值与最小值之差
            disY = y2 - y1  # 文本行左上与右上的差
            width = np.sqrt(disX * disX + disY * disY)  # 文本行的上面倾斜边的长度
            sin = disY / width
            cos = disX / width
            l_hight = y3-y1 # 文本行左边高度
            dx = np.fabs(l_hight * cos * sin)
            dy = np.fabs(l_hight * sin * sin)

            # 以放大框框的方法来修正两边的便
            if line[5] < 0:
                x1 -= dx
                y1 += dy
                x4 += dx
                y4 -= dy
            else:
                x2 += dx
                y2 += dy
                x3 -= dx
                y3 -= dy
            # fTmp0 = y3 - y1  # 文本行左边高度
            #
            #
            # # disY / width 就是倾角的sin值
            # fTmp1 = fTmp0 * disY / width
            # # disX / width就是倾角的cos值
            # x = np.fabs(fTmp1 * disX / width)  # 做补偿
            # y = np.fabs(fTmp1 * disY / width)
            # if line[5] < 0:
            #     x1 -= x
            #     y1 += y
            #     x4 += x
            #     y4 -= y
            # else:
            #     x2 += x
            #     y2 += y
            #     x3 -= x
            #     y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
