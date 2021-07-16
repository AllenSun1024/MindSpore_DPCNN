# MindSpore_DPCNN
***
## 概述：

用MindSpore框架实现DPCNN，并用于文本分类任务。
***

## 数据集：

原始数据集：
http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz

在原始数据集中，有positive和negative两类文本，每类均有5331条数据。

我们对原始数据集进行了合并，合并方式为直接读取原始数据集中解压后得到的两份文件并按行写入text_data文件。text_data文件的前5331行数据为positive文本，后5331行数据为negative文本。

开始训练前，请不要遗漏本项目中的text_data，这个合并后的数据集是我们使用的数据集。
***

## 运行流程：
1、将本项目的代码克隆到本地

2、打开train_test.py，修改train函数，具体修改内容为：将ModelCheckpoint的第二个参数directory修改为你想要保存模型的本地路径

3、运行main.py
***