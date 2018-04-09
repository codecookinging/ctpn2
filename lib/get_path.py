import os


def get_path(base, file_name=''):
    if not os.path.exists(base):
        os.makedirs(base)
    p = os.path.join(base, file_name)

    return p
