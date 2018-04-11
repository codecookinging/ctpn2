import pickle
import os
import numpy as np


def get_training_roidb(config):
    print('Preparing training data...')
    base_roidb = roidb(config)
    return base_roidb


class roidb(object):
    def __init__(self, config=None):
        print('roidb initializing......')
        assert config, 'roidb lack config'

        self.config = config
        self._image_path = config.TRAIN.TRAIN_PATH + '/Imageset'
        self._image_gt = os.path.join(config.TRAIN.TRAIN_PATH, 'Imageinfo')
        self._train_data_path = config.TRAIN.TRAIN_PATH  # E:\ctpn_yi\dataset\for_train
        self._setup()

    def _setup(self):
        self._load_image_set_index()

        self._gt_roidb()

    @property
    def roidb(self):
        return self._roidb

    def _get_image_path_with_name(self, image_name):
        image_path = os.path.join(self._image_path, image_name)
        assert os.path.exists(image_path), \
            'Image does not exist: {}'.format(image_path)
        return image_path

    '''train_set.txt
       xxxxx.jpg, width, height, channel, scale
    '''

    def _load_image_set_index(self):
        #  一个路径，里面每一行分别是 图片名，高，宽，通道数，缩放比
        image_set_file = os.path.join(self._train_data_path, 'train_set.txt')

        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        image_index = []  # 该列表存放图片名
        image_info = []  # 一个列表，列表的长度是图片的张数，列表的每个元素是一个四维向量，存放高，宽，channal，缩放比
        with open(image_set_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                image_index.append(line[0])  # 图片名
                # 这里必须是高在前，宽在后
                height = int(line[1])  # 图片的高
                wight = int(line[2])   # 图片的宽
                channal = int(line[3])  # 图片的通道数
                scale = float(line[4])  # 图片的缩放比
                image_info.append(list([height, wight, channal, scale]))
        # 一个列表，列表的长度是图片的张数，列表的每个元素是一个字符串，存放图片的名字
        self._image_index = image_index
        # 一个列表，列表的长度是图片的张数，列表的每个元素是一个四维向量，存放高，宽，channal，缩放比
        self._image_info = image_info

    def _gt_roidb(self):
        cache_file = os.path.join(self.config.TRAIN.CACHE_PATH, 'roidb.pkl')

        if os.path.exists(cache_file) and self.config.TRAIN.USE_CACHED:
            with open(cache_file, 'rb') as fid:
                gt_roidb = pickle.load(fid)
            print('gt roidb loaded from {}'.format(cache_file))

        else:
            gt_roidb = [self._process_each_image_gt(index, image_name)
                        for index, image_name in enumerate(self._image_index)]
            if not os.path.exists(self.config.TRAIN.CACHE_PATH):
                os.makedirs(self.config.TRAIN.CACHE_PATH)
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)

            print('wrote gt roidb to {}'.format(cache_file))
        self._roidb = gt_roidb
        """
        gt_roidb是一个列表，列表的每个元素是一个字典，字典的键有
        'image_name'：字符串，表示图片名字
        'boxes'：N行8列矩阵，np.int32 每一行表示一个GT,依次是左上，右上，右下，左下
        ”height“:高
        "width"：宽
        ”image_path“：图片路径
        """

    def _process_each_image_gt(self, index, image_name):

        filename = os.path.join(self._image_gt, os.path.splitext(image_name)[0] + '.txt')
        with open(filename, 'r') as f:
            gt_boxes = f.readlines()
            num_objs = len(gt_boxes)
            boxes = np.zeros((num_objs, 8), dtype=np.int32)

            single_img_info = self._image_info[index]
            for ix, box in enumerate(gt_boxes):
                try:
                    box = box.strip().split(',')
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    x3 = float(box[4])
                    y3 = float(box[5])
                    x4 = float(box[6])
                    y4 = float(box[7])
                    boxes[ix, :] = [int(x1), int(y1), int(x2), int(y2),
                                    int(x3), int(y3), int(x4), int(y4)]
                except:
                    print("the file {} has problems".format(image_name))
                    raise
        return {
            'image_path': self._get_image_path_with_name(image_name),
            'image_name': image_name,
            'height': single_img_info[0],
            'width': single_img_info[1],
            'image_scale': single_img_info[3],
            'gt_boxes': boxes,
        }
