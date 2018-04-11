import os
from lib.load_config import load_config
import sys
import cv2
import shutil
import math
import numpy as np
"""
这个脚本用来处理原始的数据，将图片按照短边600为标准进行缩放，如果缩放后长边超过1200，按照长边1200缩放，同时要缩放坐标
要将数据整理成的格式如下，存放在dataset/for_train下
以dataset/for_train为根目录
---| Imageset 保存图片文件
   | Imageinfo 保存每张图片对应的txt文本
   ----|xxxxxx.txt xxxxxx为图片名(不带扩展名)，每一行为一个文本框，格式为xmin,ymin,xmax,ymax
   ----|..........
   | train_set.txt 保存所有训练样本对应的文件名，每个占一行，格式位xxxx.jpg,width,height,channel,scale （scale为缩放比例）
### 要求在ctpn_new 目录下能够直接运行，直接读取 dataset/ICPR_text_train 下的原始训练数据进行处理，结果直接输出到到dataset下，不可人工复制粘贴，
ctpn_new/dataset 目录下的所有文件将被git忽略，以提高push速度和减少不必要的文件冲突###
"""
"""原始训练集图像存放在/dataset/ICPR_text_train/image目录下
原始训练集txt存放在/dataset/ICPR_text_train/text目录下
由原始数据集得到训练数据集调用rawdata2traindata()
"""
sys.path.append(os.getcwd())
image_dir = "E:\\ctpn_yi\\dataset\\rawImage"  # 原始训练数据集图像目录
txt_dir = "E:\\ctpn_yi\\dataset\\rawTxt"   # 原始训练数据集txt文本目录
txtfortrain_dir = "../dataset/for_train/Imageinfo"  # 保存每张图片对应的txt文本的目录
imagefortain_dir = "../dataset/for_train/Imageset"  # 保存图片文件的目录


def rawdata2traindata(config):
    # 将所有训练样本对应的文件名保存在dataset/for_train/train_set.txt 中，每个占一行,格式为xxxx.jpg, width, height, channel, scale
    # 保存图片文件，将图片按照短边600为标准进行缩放，如果缩放后长边超过1200，按照长边1200缩放，同时要缩放坐标
    # 保存每张图片对应的txt文本，每一行为一个文本框，格式为xmin,ymin,xmax,ymax
    imagedata_process(config)


def imagedata_process(config):
    # 下面两行，用于保存每张图片的信息
    filename = "train_set.txt"
    pathdir = "../dataset/for_train"
    # 判断train_set.txt是否存在，存在则删除
    if os.path.exists(pathdir + '/' + filename):
        os.remove(pathdir + '/' + filename)
    # 创建文件train_set.txt
    trainsetfile = open(pathdir + '/' + filename, 'w')
    # 得到原始文件夹下的所有文件名称
    image_files = os.listdir(image_dir)
    problem_count = 0
    for image_filename in image_files:
        imagename, ext = os.path.splitext(image_filename)        # 分离文件名和扩展名
        try:
            rawImage = cv2.imread(filename=image_dir + "/" + image_filename, flags=cv2.IMREAD_COLOR)
            img_height = rawImage.shape[0]     # 图片的高（行数）
            img_width = rawImage.shape[1]     # 图片的宽（列数）
        except:
            print("the picture {} cannot be opened".format(imagename))
            continue
        # print(img_height)
        # print(img_width)
        # 图片按照短边600为标准进行缩放,如果缩放后长边超过1000，按照长边1000缩放
        # 缩放比定义为：变形以后的图片尺寸/原图像尺寸
        minlength = min(img_width, img_height)
        maxlength = max(img_width, img_height)
        scale = 600.0/float(minlength)
        if maxlength*scale > 1000:
            scale = 1000.0/float(maxlength)

        width = int(img_width * scale)  # 缩放以后的宽
        height = int(img_height * scale)  # 缩放以后的高
        resizedImage = cv2.resize(rawImage, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # 处理GT信息, 对每张图片生成一个txt文档
        if txtdata_process(imagename, height, width, scale, resizedImage):

            train_image_path = imagefortain_dir + '/' + image_filename
            cv2.imwrite(train_image_path, resizedImage)
            # 记录缩放后的图片的信息, 高。宽，缩放比
            trainsetfile.write(image_filename + "," + str(height) + ","
                               + str(width) + "," + str(3) + "," + str(scale) + "\n")
        else:
            print("the picture {} has problems".format(imagename))
            problem_count += 1
            continue

    trainsetfile.close()
    print("the total problem pic is ", problem_count)


# 处理GT对应的txt文件
def txtdata_process(imagename, height, width, scale, resizedImage):
    gt_txt = txtfortrain_dir + '/' + imagename + ".txt"
    # 创建用于训练的txt文件
    fortraintxtfile = open(gt_txt, 'w')
    f = open(txt_dir + "/" + imagename + ".txt", 'r', encoding='UTF-8')  # 打开原始txt文件
    # iter_f = iter(f)  # 创建迭代器
    flag = True
    iter_f = f.readlines()
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        tmp1 = line.strip().split(",")  # 将原始行以“，”分割
        # exp = r",? "
        # tmp = re.split(exp, line)

        # 对原始数据按顺时针排序
        # tmp1 = clockwise_data(tmp[0:8]).reshape(-1)

        x0 = int(float(tmp1[0]) * scale)
        y0 = int(float(tmp1[1]) * scale)
        x1 = int(float(tmp1[2]) * scale)
        y1 = int(float(tmp1[3]) * scale)
        x2 = int(float(tmp1[4]) * scale)
        y2 = int(float(tmp1[5]) * scale)
        x3 = int(float(tmp1[6]) * scale)
        y3 = int(float(tmp1[7]) * scale)

        if testGT(x0, y0, x1, y1, x2, y2, x3, y3, height, width):
            fortraintxtfile.write(str(x0) + "," + str(y0) + "," + str(x1) + "," + str(y1) + ","
                                  + str(x2) + "," + str(y2) + "," + str(x3) + "," + str(y3) + "\n")
            # huizhi(resizedImage, x0, y0, x1, y1, x2, y2, x3, y3)
        else:
            flag = False
            break
    f.close()  # 关闭原始文件
    fortraintxtfile.close()  # 关闭写入文件
    if not flag:
        os.remove(gt_txt)
        return False
    return True


def huizhi(img, x0, y0, x1, y1, x2, y2, x3, y3):
    # im_path = os.path.join(imagefortain_dir, imagename + ".jpg")
    # img = cv2.imread(im_path)
    color0 = (255, 0, 0)
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)
    color3 = (255, 255, 0)
    cv2.line(img, (x0, y0), (x1, y1), color0, 2)
    cv2.line(img, (x1, y1), (x2, y2), color1, 2)
    cv2.line(img, (x2, y2), (x3, y3), color2, 2)
    cv2.line(img, (x3, y3), (x0, y0), color3, 2)
    # cv2.imwrite(os.path.join(imagefortain_dir, imagename + "_line.jpg"), img)


def testGT(x0, y0, x1, y1, x2, y2, x3, y3, height, width):
    """
    判断GT是否在图像范围内且xmin<xmax,ymin<ymax
    """
    width += 5
    height += 5
    if x0 < 0 or x0 > width:
        return False
    if x1 < 0 or x1 > width:
        return False
    if y0 < 0 or y0 > height:
        return False
    if y1 < 0 or y1 > height:
        return False
    if x2 < 0 or x2 > width:
        return False
    if x3 < 0 or x3 > width:
        return False
    if y2 < 0 or y2 > height:
        return False
    if y3 < 0 or y3 > height:
        return False
    return True


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


def clockwise_data(data_line):
    r = np.full((4, 2), 0.0, dtype='float32')
    for i in range(4):
        r[i][0] = data_line[i * 2]
        r[i][1] = data_line[i * 2 + 1]

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

    return np.array([tl, tr, br, bl], dtype="float32")


if __name__ == "__main__":
    if os.path.exists(txtfortrain_dir):
        shutil.rmtree(txtfortrain_dir)
    os.makedirs(txtfortrain_dir)

    if os.path.exists(imagefortain_dir):
        shutil.rmtree(imagefortain_dir)
    os.makedirs(imagefortain_dir)

    cfg = load_config()
    rawdata2traindata(cfg)
