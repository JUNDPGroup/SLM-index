#!/usr/bin/python
# coding=utf-8

import sys, getopt
import os
import random
import numpy as np
import tensorflow as tf

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def getUniformPoints(num, filename, dim):
    all_result = {}
    for i in range(dim - 1):
        all_result[i + 2] = []
    for i in range(num):
        node_string = ''
        for j in range(dim):
            val = random.uniform(0, 1)
            node_string = node_string + str(val) + ","
            if j >= 1:
                all_result[j + 1].append(node_string + str(i) + "\n")

    for j in range(dim - 1):
        name = filename % (num, j + 2)
        all_fo = open(name, "w")
        for i in range(num):
            all_fo.write(all_result[j + 2][i])
        all_fo.close()


def getNormalPoints(num, filename, dim):
    # 创建计算图
    graph = tf.Graph()
    with graph.as_default():
        locations_tf = []
        for i in range(dim):
            locations_tf.append(tf.random.normal([num * 2, 1], mean=0.0, stddev=1.0, dtype=tf.float32))

        with tf.compat.v1.Session(graph=graph) as sess:
            locations = sess.run(locations_tf)

    # 写入文件
    name = filename % (num, dim)
    count = 0
    with open(name, "w") as fo:
        i = 0
        while count < num and i < 2 * num:
            is_writable = True
            node_string = ''
            for j in range(dim):
                if locations[j][i][0] < 0 or locations[j][i][0] > 1:
                    is_writable = False
                    break
                node_string = node_string + str(locations[j][i][0]) + ","

            if is_writable:
                node_string = node_string + str(i) + "\n"
                fo.write(node_string)
                count += 1

            i += 1


def getSkewedPoints(num, a, filename, dim):
    graph = tf.Graph()
    with graph.as_default():
        locations_tf = [tf.random.truncated_normal([num, 1], mean=0.5, stddev=0.25, dtype=tf.float32) for _ in
                        range(dim)]
        with tf.compat.v1.Session(graph=graph) as sees:
            locations = sees.run(locations_tf)
    name = filename % (num, a, dim)
    with open(name, "w") as fo:
        for i in range(num):
            node_string = ''
            for j in range(dim - 1):
                node_string = node_string + str(locations[j][i][0]) + ","
            node_string = node_string + str(locations[dim - 1][i][0] ** a) + "," + str(i) + "\n"
            fo.write(node_string)


# def getClusterPoints(num, filename, dim, k):
#     """
#     生成符合文档定义的簇数据集：
#     - 支持生成最多100,000个点（1000个簇，每个簇100个点）
#     - 修复簇范围过小导致的重复点问题
#     """
#     if dim < 2:
#         raise ValueError("簇数据集维度必须至少为2")
#
#     # 固定每个簇的点数为100
#     points_per_cluster = 100
#
#     # 计算所需的簇数量（向上取整）
#     required_clusters = (num + points_per_cluster - 1) // points_per_cluster
#     actual_k = min(k, required_clusters)
#
#     # 最大支持1000个簇（10万个点）
#     max_possible_clusters = 1000
#     actual_k = min(actual_k, max_possible_clusters)
#     actual_num_points = actual_k * points_per_cluster
#
#     if actual_num_points != num:
#         print(f"警告：调整总点数为{actual_num_points}（{actual_k}个簇 × 每个簇{points_per_cluster}个点）")
#
#     # 生成簇中心：沿水平线等距分布，避免重叠
#     start_x = 0.05
#     end_x = 0.95
#     x_centers = np.linspace(start_x, end_x, actual_k)
#
#     centers = []
#     for x in x_centers:
#         center = [round(x, 6)] + [0.5 for _ in range(dim - 1)]  # x坐标+其他维度固定为0.5
#         centers.append(center)
#
#     # 关键修改：增大簇范围到0.0001（1e-4），确保能容纳100个六位小数的唯一值
#     cluster_size = 1e-4  # 原先是1e-6，范围太小导致无法生成唯一值
#     points = []
#     seen_points = set()
#     max_attempts_per_point = 1000  # 每个点最多尝试1000次
#     total_max_attempts = actual_num_points * max_attempts_per_point
#     current_attempts = 0
#
#     for center in centers:
#         cluster_points = 0
#         while cluster_points < points_per_cluster:
#             current_attempts += 1
#             if current_attempts > total_max_attempts:
#                 raise RuntimeError(
#                     f"无法生成足够的唯一 points（已尝试{current_attempts}次），"
#                     f"已生成{len(points)}个点，目标是{actual_num_points}个点。"
#                     f"请尝试增大cluster_size（当前为{cluster_size}）。"
#                 )
#
#             # 生成点并确保六位小数精度
#             point = []
#             for d in range(dim):
#                 # 在中心±cluster_size/2范围内采样
#                 val = random.uniform(center[d] - cluster_size / 2, center[d] + cluster_size / 2)
#                 val = max(0.0, min(1.0, val))  # 限制在[0,1]内
#                 val = round(val, 6)  # 保留六位小数
#                 point.append(val)
#
#             # 检查唯一性
#             point_tuple = tuple(point)
#             if point_tuple not in seen_points:
#                 points.append(point)
#                 seen_points.add(point_tuple)
#                 cluster_points += 1
#
#     # 写入文件
#     name = filename % (actual_num_points, actual_k, dim)
#     with open(name, "w") as fo:
#         for idx, point in enumerate(points):
#             formatted_coords = [f"{coord:.6f}" for coord in point]
#             line = ",".join(formatted_coords) + f",{idx}\n"
#             fo.write(line)
#
#     print(f"成功生成数据集：{name}，包含{actual_num_points}个点，分布在{actual_k}个簇中")


def getClusterPoints(num, filename, dim, k):
    """
    生成符合要求的簇数据集：
    - 簇中心在整个空间内随机分布（不再沿水平线分布）
    - 每个簇包含100个点
    - 支持多维数据，所有维度均随机分布
    - 所有数据精确到六位小数
    - 确保每个点都是唯一的
    """
    if dim < 2:
        raise ValueError("簇数据集维度必须至少为2")

    # 固定每个簇的点数为100
    points_per_cluster = 100

    # 计算所需的簇数量（向上取整）
    required_clusters = (num + points_per_cluster - 1) // points_per_cluster
    actual_k = min(k, required_clusters)

    # 最大支持1000个簇（10万个点）
    max_possible_clusters = 10000
    actual_k = min(actual_k, max_possible_clusters)
    actual_num_points = actual_k * points_per_cluster

    if actual_num_points != num:
        print(f"警告：调整总点数为{actual_num_points}（{actual_k}个簇 × 每个簇{points_per_cluster}个点）")

    # 关键修改：簇中心在空间内随机分布（所有维度均随机）
    # 为了避免边界问题，在[0.1, 0.9]范围内生成随机中心
    centers = []
    for _ in range(actual_k):
        # 每个维度都生成随机中心值
        center = [round(random.uniform(0.1, 0.9), 6) for _ in range(dim)]
        centers.append(center)

    # 簇范围设置为0.0001（1e-4），确保能容纳100个六位小数的唯一值
    cluster_size = 1e-4
    points = []
    seen_points = set()
    max_attempts_per_point = 1000  # 每个点最多尝试1000次
    total_max_attempts = actual_num_points * max_attempts_per_point
    current_attempts = 0

    for center in centers:
        cluster_points = 0
        while cluster_points < points_per_cluster:
            current_attempts += 1
            if current_attempts > total_max_attempts:
                raise RuntimeError(
                    f"无法生成足够的唯一 points（已尝试{current_attempts}次），"
                    f"已生成{len(points)}个点，目标是{actual_num_points}个点。"
                    f"请尝试增大cluster_size（当前为{cluster_size}）。"
                )

            # 生成点并确保六位小数精度
            point = []
            for d in range(dim):
                # 在中心±cluster_size/2范围内采样
                val = random.uniform(center[d] - cluster_size / 2, center[d] + cluster_size / 2)
                val = max(0.0, min(1.0, val))  # 限制在[0,1]内
                val = round(val, 6)  # 保留六位小数
                point.append(val)

            # 检查唯一性（多维组合检查）
            point_tuple = tuple(point)
            if point_tuple not in seen_points:
                points.append(point)
                seen_points.add(point_tuple)
                cluster_points += 1

    # 写入文件
    name = filename % (actual_num_points, actual_k, dim)
    with open(name, "w") as fo:
        for idx, point in enumerate(points):
            formatted_coords = [f"{coord:.6f}" for coord in point]
            line = ",".join(formatted_coords) + f",{idx}\n"
            fo.write(line)

    print(f"成功生成数据集：{name}，包含{actual_num_points}个点，分布在{actual_k}个簇中")

def parser(argv):
    try:
        opts, args = getopt.getopt(argv, "d:s:n:c:f:m:")
    except getopt.GetoptError:
        sys.exit(2)

    # 设置默认值
    distribution = size = skewness = dim = clusters = None

    for opt, arg in opts:
        if opt == '-d':
            distribution = arg
        elif opt == '-s':
            size = int(arg)
        elif opt == '-n':
            skewness = int(arg)
        elif opt == '-c':  # 簇数量参数
            clusters = int(arg)
        elif opt == '-f':
            filename = arg
        elif opt == '-m':
            dim = int(arg)

    # 验证必要参数
    if None in [distribution, size, filename, dim]:
        print("缺少必要参数")
        sys.exit(1)

    return distribution, size, skewness, clusters, filename, dim


if __name__ == '__main__':
    distribution, size, skewness, clusters, filename, dim = parser(sys.argv[1:])

    if distribution == 'uniform':
        filename = "datasets/uniform_%d_1_%d_.csv"
        getUniformPoints(size, filename, dim)
    elif distribution == 'normal':
        filename = "datasets/normal_%d_1_%d_.csv"
        getNormalPoints(size, filename, dim)
    elif distribution == 'skewed':
        filename = "datasets/skewed_%d_%d_%d_.csv"
        getSkewedPoints(size, skewness, filename, dim)
    elif distribution == 'cluster':
        filename = "datasets/cluster_%d_%d_%d_.csv"  # 格式：总点数_簇数量_维度
        if clusters:
            getClusterPoints(size, filename, dim, clusters)
        else:
            getClusterPoints(size, filename, dim)