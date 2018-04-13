from network.train_network import get_train_network
from ctpn.train_net import train_net
from lib.load_config import load_config
from data_process.roidb import get_training_roidb
from lib import get_path

if __name__ == '__main__':
    # print(os.getcwd())
    cfg = load_config()
    print('Using config:')
    # pprint.pprint(cfg)

    """
    @params
     use_cache 是否从重新进行data_process过程，一般dataset/for_train文件发生变化需要进行
    """
    roidb = get_training_roidb(cfg)  # 返回roidb roidb就是我们需要的对象实例

    # output_dir = get_path('dataset/output')
    # log_dir = get_path('dataset/log')

    checkpoints_dir = get_path(cfg.COMMON.CKPT)

    # print('Output will be saved to {:s}'.format(output_dir))
    # print('Logs will be saved to {:s}'.format(log_dir))

    # """
    # @params 
    # network ctpn_network 实例
    # roidb roi 列表
    # output_dir tensorflow输出的 绝对路径 要拼接本地机器所在的运行目录
    # log_dir 日志输出 绝对路径
    # max_iter 训练轮数
    # pretrain_model 预训练VGG16模型 绝对路径
    # restore bool值 是否从checkpoints断点开始恢复上次中断的训练
    # """
    vgg16_net_param = '../dataset/pretrain/VGG_imagenet.npy'
    network = get_train_network(cfg)
    train_net(cfg, network, roidb, checkpoints_dir, max_iter=cfg.TRAIN.MAX_ITER,
              pretrain_model=vgg16_net_param, restore=True)
