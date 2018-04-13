#coding:utf-8
import tensorflow as tf
import numpy as np
import shutil
import os
import cv2
from lib.timer import Timer
from lib.text_connector.detectors import TextDetector
from exceptions import NoPositiveError


class TestClass(object):
    def __init__(self, cfg, network):
        self._cfg = cfg
        self._net = network

    # 画方框,被ctpn()调用
    def draw_boxes(self, img, image_name, boxes):
        """
        :param img: 最原始的图片矩阵
        :param image_name: 图片地址
        :param boxes: N×9的矩阵，表示N个拼接以后的完整的文本框。
        每一行，前八个元素一次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
        :return:
        """
        # base_name = image_name.split('/')[-1]
        base_name = os.path.basename(image_name)
        b_name, ext = os.path.splitext(base_name)

        with open(os.path.join(self._cfg.TEST.RESULT_DIR_TXT, '{}.txt'.format(b_name)), 'w') as f:
            for box in boxes:
                # TODO 下面注释掉的这两行不知是啥意思
                # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                #     continue
                # 默认用红色线条绘制，可能性最低
                color = (0, 0, 255)  # 颜色为BGR
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
                cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
                # cv2.putText(img, 'score:{}'.format(box[8]), (int(box[0]), int(box[1])),
                #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                x4 = box[4]
                y4 = box[5]
                x3 = box[6]
                y3 = box[7]

                line = ','.join([str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4)]) + '\n'
                f.write(line)

        cv2.imwrite(os.path.join(self._cfg.TEST.RESULT_DIR_PIC, base_name), img)

    # 改变图片的尺寸，被ctpn()调用
    @ staticmethod
    def resize_im(im, scale, max_scale=None):
        # 缩放比定义为 修改后的图/原图
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    # 被test_net()调用
    def ctpn(self, sess, net, image_name):
        """
        :param sess: 会话
        :param net: 创建的测试网络
        :param image_name: 所要测试的单张图片的目录
        :return:
        """


        # 读取图片
        image = cv2.imread(image_name)
        # shape = image.shape[:2]  # 获取高，宽
        # resize_im，返回缩放后的图片和相应的缩放比。缩放比定义为 修改后的图/原图
        img, scale = TestClass.resize_im(image, scale=self._cfg.TEST.SCALE, max_scale=self._cfg.TEST.MAX_SCALE)
        shape = img.shape[:2]  # 获取缩放后的高，宽
        # 将图片去均值化
        im_orig = img.astype(np.float32, copy=True)
        im_orig -= self._cfg.TRAIN.PIXEL_MEANS

        # 将缩放和去均值化以后的图片，放入网络进行前向计算，获取分数和对应的文本片段，该片段为映射到最原始图片的坐标
        scores, boxes = TestClass.test_ctpn(sess, net, im_orig, scale)

        # 此处调用了一个文本检测器
        textdetector = TextDetector(self._cfg)
        """
        输入参数分别为：
        N×4矩阵，每行为一个已经映射回最初的图片的文字片段坐标
        N维向量，对应的分数
        两维向量，分别为最原始图片的高宽
        返回：
        一个N×9的矩阵，表示N个拼接以后的完整的文本框。
        每一行，前八个元素一次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
        """
        # 缩放后的boxes
        boxes = textdetector.detect(boxes, scores, shape)
        boxes[:, 0:8] = boxes[:, 0:8] / scale
        # 在原始图片上画图
        self.draw_boxes(image, image_name, boxes)


    def test_net(self, graph):

        timer = Timer()
        timer.tic()

        if os.path.exists(self._cfg.TEST.RESULT_DIR_TXT):
            shutil.rmtree(self._cfg.TEST.RESULT_DIR_TXT)
        os.makedirs(self._cfg.TEST.RESULT_DIR_TXT)

        if os.path.exists(self._cfg.TEST.RESULT_DIR_PIC):
            shutil.rmtree(self._cfg.TEST.RESULT_DIR_PIC)
        os.makedirs(self._cfg.TEST.RESULT_DIR_PIC)

        saver = tf.train.Saver()
        # 创建一个Session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 不能太大，否则报错

        sess = tf.Session(config=config, graph=graph)

        # 获取一个Saver()实例

        # 恢复模型参数
        ckpt = tf.train.get_checkpoint_state(self._cfg.COMMON.CKPT)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
            print('done')
        else:
            raise 'Check your pretrained {:s}'.format(self._cfg.TEST.RESULT_DIR)

        # # TODO 这里需要仔细测试一下
        # im_names = glob.glob(os.path.join(self._cfg.TEST.DATA_DIR, '*.png')) + \
        #            glob.glob(os.path.join(self._cfg.TEST.DATA_DIR, '*.jpg'))

        im_names = os.listdir(self._cfg.TEST.DATA_DIR)

        assert len(im_names) > 0, "Nothing to test"
        i = 0
        for im in im_names:
            im_name = os.path.join(self._cfg.TEST.DATA_DIR, im)
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(('Testing for image {:s}'.format(im_name)))
            try:
                self.ctpn(sess, self._net, im_name)
            except NoPositiveError:
                print("Warning!!, get no region of interest in picture {}".format(im))
                continue
            except:
                print("the pic {} may has problems".format(im))
                continue
            i += 1
            if i % 10 == 0:
                timer.toc()
                print('Detection took {:.3f}s for 10 pic'.format(timer.total_time))

        # 最后关闭session
        sess.close()

    @ staticmethod
    def test_ctpn(sess, net, im, scale):
        # print("===============", type(im))
        im_info = np.array([im.shape[0], im.shape[1], scale])
        im = im[np.newaxis, :]
        feed_dict = {net.data: im, net.im_info: im_info, net.keep_prob: 1.0}
        fetches = [net.get_output('rois'), ]

        # (1 x H x W x A, 5) 第一列为正例的概率,已经排序了，后四列为映射回输入图片的，经过回归修正以后的，预测的文本片段坐标
        # 已经经过了非极大值抑制！！！！！！！！！！！！！
        rois = sess.run(fetches=fetches, feed_dict=feed_dict)
        if len(rois) == 0:
            raise NoPositiveError("Found no region of interest")
        # sess.run是以列表形式返回结果，所以这里需要[0]，以取出数组
        rois = rois[0]
        scores = rois[:, 0]
        # 这里是缩放后的坐标
        boxes = rois[:, 1:5]
        return scores, boxes
