from dataset import mindrecord_dataset
from preprocess import PreProcessor
from model import DPCNN
from mindspore import Tensor

'''数据预处理'''
data_processor = PreProcessor(mindrecord_dataset)
data_processor.text_tokenization()
data_processor.build_vocabulary()
data_processor.word_to_idx()
data_processor.padding_sentence()
data_processor.shuffle_dataset()
data_processor.set_batch()


model = DPCNN()
for item in data_processor.dataset.create_dict_iterator(output_numpy=True):
    x = model(Tensor(item["index"]))
    break
