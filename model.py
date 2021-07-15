import mindspore.nn as nn
import mindspore.ops as ops

class DPCNN(nn.Cell):
    def __init__(self):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size=8002, embedding_size=300, padding_idx=0)
        self.conv_region = nn.Conv2d(in_channels=1, out_channels=250, kernel_size=(3, 300), stride=1, pad_mode="valid")
        self.conv = nn.Conv2d(in_channels=250, out_channels=250, kernel_size=(3, 1), stride=1, pad_mode="valid")
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.relu = nn.ReLU()
        self.padding1 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0)), mode="CONSTANT")
        self.padding2 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 0)), mode="CONSTANT")
        self.expend_dims = ops.ExpandDims()   # unsqueeze()
        self.squeeze = ops.Squeeze(axis=(2, 3))
        self.fc = nn.Dense(in_channels=250, out_channels=2)  # nn.Linear

    def construct(self, x):
        x = self.embedding(x)
        x = self.expend_dims(x, 1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.shape[2] > 2:
            x = self._block(x)
        x = self.squeeze(x)
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        relu2 = ops.ReLU()
        x = relu2(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = relu2(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


