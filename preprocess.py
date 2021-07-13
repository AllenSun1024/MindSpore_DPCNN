import mindspore.dataset.text as text
import mindspore.dataset.transforms.c_transforms as ct

class PreProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vocabulary = None

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

    def shuffle_dataset(self):
        self.dataset = self.dataset.shuffle(buffer_size=100)
