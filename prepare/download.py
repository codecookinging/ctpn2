import hashlib
import requests
import os
import zipfile
import shutil


def md5sum(filename):
    fd = open(filename, "rb")
    fcont = fd.read()
    fd.close()
    fmd5 = hashlib.md5(fcont)
    return fmd5.hexdigest()


def get_file_path(base, file_name):
    file_path = os.path.join(base, file_name)
    if not os.path.exists(base):
        os.makedirs(base)

    return file_path


def un_zip(file_name, extract_dir):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(extract_dir):
        pass
    else:
        os.mkdir(extract_dir)
    for names in zip_file.namelist():
        zip_file.extract(names, extract_dir)
    zip_file.close()


def main():
    file_list = [
        'VGG_imagenet.npy',
        'icpr_text_train_10000.zip'
    ]
    url_list = [
        'http://ch-10035517.cossh.myqcloud.com/VGG_imagenet.npy',
        'http://ch-10035517.cossh.myqcloud.com/icpr_text_train_10000.zip'
    ]
    base = 'dataset'

    md5 = [
        '102f510d020773a884e76814e197170f',  # vgg16 md5
        'e7ea68b7d69b248c98328a590dc82839'
    ]
    prefix = ['pretrain/', 'tmp/']

    for ix, url in enumerate(url_list):
        path = get_file_path(os.path.join(base, prefix[ix]), file_list[ix])
        if os.path.exists(path) and md5sum(path) == md5[ix]:
            print('using exits file {}'.format(file_list[ix]))
        else:
            print('starting download {}'.format(file_list[ix]))
            r = requests.get(url)
            print('download file {} successful'.format(file_list[ix]))
            with open(path, "wb") as code:
                code.write(r.content)
            print('write {} file successful'.format(file_list[ix]))
        if ix == 1:
            print('starting extracting train data......')
            extract_path = get_file_path(base, 'ICPR_text_train')
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
            os.makedirs(extract_path)

            un_zip(path, extract_path)
            print('extracting over')
            img_len = len(os.listdir(extract_path + '/image_10000'))
            txt_len = len(os.listdir(extract_path + '/text_10000'))

            # print("image size", img_len)
            # print("text size", txt_len)

            os.rename(extract_path + '/image_10000', extract_path + '/image')
            os.rename(extract_path + '/text_10000', extract_path + '/text')
            print('starting cleaning tmp file')
            os.listdir()
            shutil.rmtree(os.path.join(extract_path, '__MACOSX'))
            shutil.rmtree(os.path.join(base, prefix[ix]))


main()
