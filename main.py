from dataset import mindrecord_dataset
from preprocess import PreProcessor
from model import DPCNN
from train_test import train, test
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

'''数据预处理'''
data_processor = PreProcessor(mindrecord_dataset)
data_processor.text_tokenization()
data_processor.build_vocabulary()
data_processor.word_to_idx()
data_processor.padding_sentence()
data_processor.create_new()
data_processor.shuffle_dataset()
data_processor.split_dataset()
data_processor.set_batch()

'''获取划分后的数据集'''
train_dataset = data_processor.train_data
test_dataset = data_processor.test_data

'''训练模型'''
model = DPCNN()
check_file_path = train(model, train_dataset)

'''测试模型'''
model = DPCNN()
test(model, test_dataset, check_file_path)
