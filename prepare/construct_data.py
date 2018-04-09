import os
from lib.load_config import load_config
import sys
import cv2
import shutil
import re
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
image_dir = "../ICDAR2013/image"  # 原始训练数据集图像目录
txt_dir = "../ICDAR2013/gt"   # 原始训练数据集txt文本目录
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
    for image_filename in image_files:
        imagename, ext = os.path.splitext(image_filename)        # 分离文件名和扩展名
        rawImage = cv2.imread(filename=image_dir + "/" + image_filename, flags=cv2.IMREAD_COLOR)

        img_height = rawImage.shape[0]     # 图片的高（行数）
        img_width = rawImage.shape[1]     # 图片的宽（列数）
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

        # 处理GT信息, 对每张图片生成一个txt文档
        resizedImage = cv2.resize(rawImage, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if txtdata_process(imagename, height, width, scale, resizedImage):

            train_image_path = imagefortain_dir + '/' + image_filename
            cv2.imwrite(train_image_path, resizedImage)
            # 记录缩放后的图片的信息, 高。宽，缩放比
            trainsetfile.write(image_filename + "," + str(height) + ","
                               + str(width) + "," + str(3) + "," + str(scale) + "\n")

    trainsetfile.close()


def txtdata_process(imagename, height, width, scale, resizedImage):
    gt_txt = txtfortrain_dir + '/' + imagename + ".txt"
    # 创建用于训练的txt文件
    fortraintxtfile = open(gt_txt, 'w')
    f = open(txt_dir + "/" + imagename + ".txt", 'r', encoding='UTF-8')  # 打开原始txt文件
    iter_f = iter(f)  # 创建迭代器
    flag = True
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        # tmp = line.split(",")  # 将原始行以“，”分割
        exp = r",? "
        tmp = re.split(exp, line)
        x0 = float(tmp[0])*scale
        y0 = float(tmp[1])*scale
        x1 = float(tmp[2])*scale
        y1 = float(tmp[3])*scale

        if testGT(x0, y0, x1, y1,  height, width):
            fortraintxtfile.write(str(x0) + "," + str(y0) + "," + str(x1) + "," + str(y1) + "\n")
            # huizhi(resizedImage, x0, y0, x1, y1)
        else:
            flag = False
            break
    f.close()  # 关闭原始文件
    fortraintxtfile.close()  # 关闭写入文件
    if not flag:
        os.remove(gt_txt)
        return flag
    return flag


def huizhi(img, xmin, ymin, xmax, ymax):
    # im_path = os.path.join(imagefortain_dir, imagename + ".jpg")
    # img = cv2.imread(im_path)
    color = (255, 0, 0)
    cv2.line(img, (int(xmin), int(ymin)), (int(xmin), int(ymax)), color, 2)
    cv2.line(img, (int(xmin), int(ymax)), (int(xmax), int(ymax)), color, 2)
    cv2.line(img, (int(xmax), int(ymax)), (int(xmax), int(ymin)), color, 2)
    cv2.line(img, (int(xmax), int(ymin)), (int(xmin), int(ymin)), color, 2)
    # cv2.imwrite(os.path.join(imagefortain_dir, imagename + "_line.jpg"), img)


def testGT(xmin, ymin, xmax, ymax, height, width):
    """
    判断GT是否在图像范围内且xmin<xmax,ymin<ymax
    """
    if xmin < 0 or xmin > width:
        return False
    if xmax < 0 or xmax > width:
        return False
    if ymin < 0 or ymin > height:
        return False
    if ymax < 0 or ymax > height:
        return False
    return True


if __name__ == "__main__":
    if os.path.exists(txtfortrain_dir):
        shutil.rmtree(txtfortrain_dir)
    os.makedirs(txtfortrain_dir)

    if os.path.exists(imagefortain_dir):
        shutil.rmtree(imagefortain_dir)
    os.makedirs(imagefortain_dir)

    cfg = load_config()
    rawdata2traindata(cfg)
