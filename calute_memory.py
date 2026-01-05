# coding=utf-8
DATA_K = int(input("K:"))
DATA_D = int(input("数据集维度:"))
DATA_NUM = int(input("数据集大小(w): "))
CLASS_NUM = int(input("分类总数量: "))
RES_NUM = int(input("回归网络数量: "))

LEAFNODE_NUM = int(input("桶数量: "))
RSMI_LEAFNODE_NUM = int(input("RSMI叶节点数量: "))
RSMI_NOLEAFNODE_NUM = int(input("RSMI非叶节点数量: "))

DATA_NUM *= 10000
RSMI_NODE_NUM = RSMI_NOLEAFNODE_NUM + RSMI_LEAFNODE_NUM
NODE_NUM = RSMI_NODE_NUM + LEAFNODE_NUM

#计算回归参数 叶回归+非叶回归
RES_NET =RES_NUM  * (50 * (DATA_D + 1) + 1)

#计算分类参数
CLASS_NET = 50 * (DATA_D + CLASS_NUM)

#计算mbr 两个数据点
MBR = 2 * NODE_NUM * DATA_D

#计算数据点
DATA = DATA_NUM * DATA_D

#计算存储开销 2*1024=2048位=256字节
MEMORY_SIZE = (RES_NET + CLASS_NET + MBR + DATA)*((2*DATA_K)/8)/1024/1024

# 输出结果
print("结果是: {}MB".format(MEMORY_SIZE))