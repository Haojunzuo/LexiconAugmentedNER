import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import re

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 清洗数据：将数据中的非汉字去掉
def clear_data(data):
    if len(data) == 0:
        return data
    reslut = ""
    for item in data:
        if '\u4e00' <= item <= '\u9fff':
            reslut += item
    return reslut


# 将数据转换为dict
def make_char_dicts(data):
    char_dict = {}
    cnt = 0
    for item in data:
        if item not in char_dict.keys():
            char_dict[item] = cnt
            cnt += 1
    return char_dicts


# 平均特征长度：6
def process_data(file_name):
    file = open(file_name, 'r')
    data_list = file.read().split('\n')  # 读取文本
    data_dicts = {}  # 创建一个dict  key：char   value：radical
    char_dicts = {}  # 创建一个dict  key：char   value：index
    cnt = 0
    for data in data_list:  # 对每一项遍历
        data = data.strip().split('\t')  # 获取每一项文本并去掉首尾空格
        if len(data) > 1:
            char = data[0]
            bushou = clear_data(re.findall(".*=(.*)", data[1]))
            jianti = clear_data(re.findall(".*=(.*)", data[2]))
            fanti = clear_data(re.findall(".*=(.*)", data[3]))
            gouzao = clear_data(re.findall(".*=(.*)", data[4]))
            shouwei = clear_data(re.findall(".*=(.*)", data[5]))
            radical = bushou + jianti + fanti + gouzao + shouwei
            data_dicts[char] = radical

            if char not in char_dicts.keys():
                char_dicts[char] = cnt
                cnt += 1
            for item in radical:
                if item not in char_dicts.keys():
                    char_dicts[item] = cnt
                    cnt += 1

    return data_dicts, char_dicts


# data_dicts, char_dicts = process_data("./radical.dict")
# # for key in char_dicts:
# #     print(key+':'+str(char_dicts[key]))
# for key in data_dicts:
#     print(key + ':' + data_dicts[key])


# 随机初始化向量  从uni集中建立dict
def make_uni_dict(file_name):
    file = open(file_name, 'r')
    data_list = file.read().split('\n')  # 读取文本
    uni_dicts = {}  # 创建一个dict  key：char   value:index
    cnt = 0
    for data in data_list:  # 对每一项遍历
        data = data.strip().split()  # 获取每一项文本并去掉首尾空格
        if len(data) > 0:
            uni_dicts[data[0]] = cnt
            cnt += 1
    return uni_dicts

# 从gigaword_chn.all.a2b.uni.ite50.vec中提取预训练的字向量
def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim


def save_radical_embedding(embedd_dict, file_name):
    data_len = 11327  # 数据长度
    with open(file_name, 'w') as f:
        for key in embedd_dict:
            f.write(key + " ")  # 保存字符
            value = embedd_dict[key]
            # 保存每个字符的50d张量
            for j in range(50):
                f.write(str(np.around(value[0][j], 6)) + " ")
            f.write('\n')

if __name__ == '__main__':

    data_dicts, char_dicts = process_data("./radical.dict")
    uni_dicts = make_uni_dict("../data/tencent-ailab-embedding-zh-d200-v0.2.0-s-c.txt")
    embedd_dict, embedd_dim = load_pretrain_emb('../data/tencent-ailab-embedding-zh-d200-v0.2.0-s-c.txt')


    # 遍历 char - radical字典
    for key in data_dicts:
        char = key  # 字符
        radical = data_dicts[char]  # 对应部首
        radical_len = len(radical)  # 部首分解的长度
        # 为每个字符 创建一个 6 * embedd_dim的张量
        radical_tensor = torch.randn(size=(6, embedd_dim), layout=torch.strided, device=device, requires_grad=False)
        # 对于每一个字符 此操作相当于截断
        for idx in range(min(6, radical_len)):
            radical_item = radical[idx]  # 每一个部首
            # 判断是否在uni中
            if radical_item in uni_dicts.keys():
                radical_tensor[idx] = torch.tensor(embedd_dict[radical_item])

        """
            in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度；图像中为RGB三通道。
            out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
            kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小
            stride(int or tuple, optional) - 卷积步长
            dilation(int or tuple, `optional``) – 卷积核元素之间的间距
        """

        # 不求导
        with torch.no_grad():
            # 做卷积   在文本应用中，channel即为词向量的维度
            embedding_dim, kernel_sizes, num_channels = embedd_dim, 3, embedd_dim  # 定义多组卷积核
            # 升维度  [radical_num , embedding dim] --> [batch_size,radical_num , embedding dim]
            radical_tensor = radical_tensor.unsqueeze(0)
            # [1,6,50] --> [1,50,6]  6个部首，每个部首长度embedd_dim，改变其形状
            radical_tensor = radical_tensor.permute(0, 2, 1)
            # 卷积函数定义
            text_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_channels,
                                  kernel_size=kernel_sizes, stride=1).cuda(1)
            conv_radical = text_conv(radical_tensor)  # 卷积后结果
            result = F.max_pool1d(conv_radical, kernel_size=4)  # 最大池化
            result = result.permute(0, 2, 1)  # 还原形状
            # 降维度  [batch_size,radical_num , embedding dim] --> [radical_num , embedding dim]
            result = result.squeeze(0)
        # 更新radical embedding
        embedd_dict[char] = result.cpu().numpy()
    save_radical_embedding(embedd_dict, "../data/max_pool_radical_1219.vec")