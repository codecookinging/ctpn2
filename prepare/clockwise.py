import os
import math
import csv
import shutil
import numpy as np
import pandas as pd


def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


if __name__ == '__main__':
    # 原数据集的txt文件目录
    in_path = "E:/alidata/ICPR_text_train_part2_20180313/err"
    # 生成新txt文件目录
    out_path = "E:/alidata/ICPR_text_train_part2_20180313/err_refresh"
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    dirs = os.listdir(in_path)
    for i in dirs:
        try:
            if os.path.splitext(i)[1] == ".txt":
                data = pd.read_csv(os.path.join(in_path, i), header=None, quoting=csv.QUOTE_NONE, encoding='utf8').iloc[:, :8].values
                with open(os.path.join(out_path, i), 'w') as f:
                    for index, d1 in enumerate(data):
                        r = np.full((4, 2), 0.0, dtype='float32')
                        for j in range(4):
                            r[j][0] = d1[j * 2]
                            r[j][1] = d1[j * 2 + 1]

                        xSorted = r[np.argsort(r[:, 0]), :]
                        leftMost = xSorted[:2, :]
                        rightMost = xSorted[2:, :]

                        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
                        (tl, bl) = leftMost

                        vector_0 = np.array(bl - tl)
                        vector_1 = np.array(rightMost[0] - tl)
                        vector_2 = np.array(rightMost[1] - tl)

                        angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
                        (br, tr) = rightMost[np.argsort(angle), :]

                        f.write(','.join(list(map(str, (tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1])))) + '\n')

        except :
            print(i + " Error")


    print("success")
