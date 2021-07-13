import mindspore.nn as nn

class DPCNN(nn.cell):
    def __init__(self):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size=8002, embedding_size=300, padding_idx=0)
        self.conv_region = nn.Conv2d(in_channels=1, out_channels=250, kernel_size=(3, 300), stride=1)
        self.conv = nn.Conv2d(in_channels=250, out_channels=250, kernel_size=(3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.Pad(paddings=((1, 1), (2, 2)), mode="CONSTANT")

