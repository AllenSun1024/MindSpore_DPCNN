import os
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds

print("\n开始构造数据集...")

'''将原始数据转换为mindrecord'''

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
        line = f.readline()
        sample = {"text": line, "label": 1}
        data.append(sample)
        writer.write_raw_data(data)
    for i in range(5331):
        data = []
        line = f.readline()
        sample = {"text": line, "label": 0}
        data.append(sample)
        writer.write_raw_data(data)
writer.commit()


'''加载mindrecord数据集'''
DATA_FILE = ["text_label.mindrecord"]

mindrecord_dataset = ds.MindDataset(DATA_FILE)

print("数据集构造完毕!\n")