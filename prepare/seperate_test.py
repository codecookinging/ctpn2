import os
import shutil
import random
import sys

sys.path.append(os.getcwd())
from lib.get_path import get_path

path = 'dataset/ICPR_text_train'
ims = os.listdir(os.path.join(path, 'image'))
txts = os.listdir(os.path.join(path, 'text'))

random.shuffle(ims)

for im in ims[:1000]:
    image_to = get_path(os.path.join('dataset/for_test', 'image'))
    txt_to = get_path(os.path.join('dataset/for_test', 'txt'))

    if os.path.isfile(os.path.join(path, 'image', im)):
        shutil.move(os.path.join(path, 'image', im), image_to)
        shutil.move(os.path.join(path, 'text', os.path.splitext(im)[0] + '.txt'), txt_to)
