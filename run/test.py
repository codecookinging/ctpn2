from network.test_network import get_test_network
from lib.load_config import load_config
from ctpn.test_net import TestClass
import tensorflow as tf

if __name__ == "__main__":
    # 加载配置文件
    cfg = load_config()
    # pprint.pprint(cfg)
    with tf.Graph().as_default() as g:
        # 获取测试网络， 一个空网络
        network = get_test_network(cfg)
        # saver = tf.train.Saver()
        # 获取测试类实例，这时候也还没有把参数填写进去
        testclass = TestClass(cfg, network)
        # 开始测试
        testclass.test_net(g)

