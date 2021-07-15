import os
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds

'''将原始数据转换为mindrecord'''
print("\n开始构造数据集...")

MINDRECORD_FILE = "text_label.mindrecord"
if os.path.exists(MINDRECORD_FILE):
    os.remove(MINDRECORD_FILE)
    os.remove(MINDRECORD_FILE + ".db")
writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
schema = {
    "text": {"type": "string"},
    "label": {"type": "int32"}
}
writer.add_schema(schema)
writer.add_index(["text", "label"])

with open("text_data", "r") as f:
    for i in range(5331):
        data = []
        line = f.readline().strip()
        sample = {"text": line, "label": 1}
        data.append(sample)
        writer.write_raw_data(data)
    for i in range(5331):
        data = []
        line = f.readline().strip()
        sample = {"text": line, "label": 0}
        data.append(sample)
        writer.write_raw_data(data)
writer.commit()

'''加载mindrecord数据集'''
DATA_FILE = ["text_label.mindrecord"]
global mindrecord_dataset
mindrecord_dataset = ds.MindDataset(DATA_FILE)

print("数据集构造完毕!\n")


'''基于mindrecord_dataset实现可迭代的数据集类'''
class NewDataset:
    def __init__(self, raw_dataset):
        self.datas = []
        self.labels = []
        for item in raw_dataset.create_dict_iterator(output_numpy=True):
            x = item["index"]
            y = item["label"]
            self.datas.append(x)
            self.labels.append(y)

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)
