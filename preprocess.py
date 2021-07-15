import mindspore.dataset.text as text
import mindspore.dataset.transforms.c_transforms as ct
from dataset import NewDataset
import mindspore.dataset as ds

class PreProcessor:
    def __init__(self, dataset):
        print("\n开始数据预处理...")
        self.dataset = dataset
        self.vocabulary = None
        self.train_data = None
        self.test_data = None

    def text_tokenization(self):
        tokenize_op = text.WhitespaceTokenizer()
        self.dataset = self.dataset.map(operations=tokenize_op, input_columns=["text"])

    def build_vocabulary(self):
        self.vocabulary = self.dataset.build_vocab(columns=["text"], freq_range=(2, 10000), top_k=8000, special_tokens=["<pad>", "<unk>"], special_first=True)

    def word_to_idx(self):
        look_up = text.Lookup(self.vocabulary, unknown_token="<unk>")
        self.dataset = self.dataset.map(operations=look_up, input_columns=["text"], output_columns=["index"])

    def padding_sentence(self):
        self.dataset = self.dataset.map(operations=ct.Slice(slice(0, 32)), input_columns=["index"])
        self.dataset = self.dataset.map(operations=ct.PadEnd(pad_shape=[32], pad_value=0), input_columns=["index"])

    def create_new(self):
        dataset_generator = NewDataset(self.dataset)
        self.dataset = ds.GeneratorDataset(dataset_generator, ["index", "label"], shuffle=True)

    def shuffle_dataset(self):
        self.dataset = self.dataset.shuffle(buffer_size=self.dataset.get_dataset_size())

    def split_dataset(self):
        self.train_data, self.test_data = self.dataset.split(sizes=[0.8, 0.2], randomize=True)

    def set_batch(self):
        self.train_data = self.train_data.batch(batch_size=32, drop_remainder=True)
        self.train_data = self.train_data.repeat(count=1)
        self.test_data = self.test_data.batch(batch_size=32, drop_remainder=True)
        self.test_data = self.test_data.repeat(count=1)
        print("数据预处理完毕!\n")
