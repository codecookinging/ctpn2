import yaml
from easydict import EasyDict as edict


def load_config():
    config_path = 'E:\ctpn_yi\configure.yml'
    with open(config_path, 'r', encoding='UTF-8') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg
