from mindspore import Model
import mindspore.nn as nn
from mindspore.nn import Accuracy
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def train(model, train_dataset):
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.001)
    model = Model(model, criterion, optimizer, {'acc': Accuracy()})
    print("\n开始训练模型...")
    config = CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size() * 2,
                              keep_checkpoint_max=5)
    check_point = ModelCheckpoint(prefix="DPCNN",
                                  directory="/home/ubuntu/disk2/AllenSun_Projects/DPCNN_MindSpore/check_point",
                                  config=config)
    loss_monitor = LossMonitor()
    time_monitor = TimeMonitor(data_size=train_dataset.get_dataset_size())
    model.train(epoch=10, train_dataset=train_dataset,
                callbacks=[time_monitor, loss_monitor, check_point], dataset_sink_mode=False)
    print("模型训练完毕!\n")


def test(model, test_dataset):
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model_new = Model(model, criterion, metrics={
        'acc': Accuracy()
    })
    print("\n开始测试模型...")
    params = load_checkpoint("/home/ubuntu/disk2/AllenSun_Projects/DPCNN_MindSpore/check_point/DPCNN-1_266.ckpt")
    load_param_into_net(model, params)
    acc = model_new.eval(test_dataset, dataset_sink_mode=False)
    print("模型测试准确率：{}\n".format(acc))
