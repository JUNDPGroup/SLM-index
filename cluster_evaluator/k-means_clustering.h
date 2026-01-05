#ifndef KMEANS_CLUSTERING_H
#define KMEANS_CLUSTERING_H

#include "../entities/Point.h"
#include <vector>
#include <unordered_map>

// 聚类结果结构体
struct KMeansResult {
    // std::unordered_map<int, std::vector<Point>> clustered_points; // 簇ID到点集的映射
    std::map<int, std::vector<Point>> clustered_points;
    std::vector<Point> centroids;                       // 最终质心
    int k;                                              // 簇个数
    double silhouette_score;                            // 轮廓系数得分
};

/**
 * @brief 执行K-means聚类的主接口函数
 * 
 * @param points 输入点集
 * @param k 指定聚类个数，0表示自动确定最优k值
 * @param max_k 自动确定k时的最大k值
 * @param min_k 自动确定k时的最小k值
 * @param max_iters 聚类最大迭代次数
 * @param trials_per_k 每个k值尝试次数
 * @return KMeansResult 聚类结果
 */
KMeansResult perform_kmeans_clustering(
    std::vector<Point>& points, 
    int k = 0,
    int max_k = 10,
    int min_k = 2,
    int max_iters = 100,
    int trials_per_k = 3
);

#endif // KMEANS_CLUSTERING_H