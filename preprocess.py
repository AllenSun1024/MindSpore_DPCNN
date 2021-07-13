from dataset import mindrecord_dataset
import mindspore.dataset.text as text


class PreProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vocabulary = None

    def text_tokenization(self):
        tokenize_op = text.WhitespaceTokenizer()
        self.dataset = self.dataset.map(operations=tokenize_op, input_columns=["text"])

    # def build_vocabulary(self):
    #     self.vocabulary = self.dataset.build_vocab(columns=["text"], freq_range=(2, 10000), top_k=7116, special_tokens=["<pad>", "<unk>"], special_first=True)

t_op = text.WhitespaceTokenizer()
mindrecord_dataset = mindrecord_dataset.map(operations=t_op, input_columns=["text"])
# vocab = mindrecord_dataset.build_vocab(columns=["text"], freq_range=(2, 10000), top_k=7116, special_tokens=["<pad>", "<unk>"], special_first=True)
vocab = text.Vocab.from_dataset(dataset=mindrecord_dataset, columns=["text"], freq_range=(2, 10000), top_k=7000, special_tokens=["<pad>", "<unk>"], special_first=True)
print(vocab)
