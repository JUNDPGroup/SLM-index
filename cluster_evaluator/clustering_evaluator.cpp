#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>
#include <iomanip>
#include <flann/flann.hpp>
#include <limits>
#include "../entities/Point.h"

// Eigen包含在实现文件中，避免影响其他编译单元
#include <eigen3/Eigen/Dense>
#include "clustering_evaluator.h"
// mt19937 gen{42};
namespace ClusterEval {
    using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixType = EigenMatrix;
}

// 辅助函数声明 - 全部在实现文件中
static ClusterEval::MatrixType pointsToMatrix(const std::vector<Point>& points);
static double hopkinsStatistic(const ClusterEval::MatrixType& X, double sampleSize = 0.3, int randomState = 42);
static std::map<std::string, double> analyzeNeighborDistances(const ClusterEval::MatrixType& X, int k = 5);
static std::pair<std::map<std::string, std::string>, int> evaluateClusteringSuitability(
    double hopkins, 
    const std::map<std::string, double>& distStats
);

// // 将Point向量转换为Eigen矩阵
// static ClusterEval::MatrixType pointsToMatrix(const std::vector<Point>& points) {
//     if (points.empty()) {
//         return ClusterEval::MatrixType(0, 0);
//     }
    
//     size_t numPoints = points.size();
//     ClusterEval::MatrixType X(numPoints, 2);  // 固定为2维
    
//     for (size_t i = 0; i < numPoints; ++i) {
//         X(i, 0) = points[i].x;
//         X(i, 1) = points[i].y;
//     }
    
//     return X;
// }

// // 将Point向量转换为Eigen矩阵
static ClusterEval::MatrixType pointsToMatrix(const std::vector<Point>& points) {
    if (points.empty()) {
        return ClusterEval::MatrixType(0, 0);
    }

    size_t numPoints = points.size();
    int dim = points[0].dim;
    ClusterEval::MatrixType X(numPoints, dim);  // 动态维度

    for (size_t i = 0; i < numPoints; ++i) {
        for (int j = 0; j < dim; ++j) {
            X(i, j) = points[i].coords[j];
        }
    }

    return X;
}

// // 霍普金斯统计量计算
// static double hopkinsStatistic(const ClusterEval::MatrixType& X, double sampleSize, int randomState) {
//     int n = X.rows();
//     if (n < 5) {
//         std::cout << "警告: 样本量过小，霍普金斯统计量可能不可靠" << std::endl;
//         return 0.5;
//     }
    
//     int m = (sampleSize > 0) ? static_cast<int>(sampleSize * n) : std::min(50, n);
//     m = std::max(1, m); // 至少取1个样本
    
//     // 转换数据为FLANN兼容格式
//     std::vector<float> dataVec;
//     dataVec.reserve(n * X.cols());
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < X.cols(); j++) {
//             dataVec.push_back(static_cast<float>(X(i, j)));
//         }
//     }
    
//     // 构建FLANN索引进行最近邻搜索
//     flann::Matrix<float> flannData(dataVec.data(), n, X.cols());
//     flann::Index<flann::L2<float>> index(flannData, flann::KDTreeSingleIndexParams(10));
//     index.buildIndex();
    
//     // 生成均匀分布的随机点
//     std::mt19937 gen(randomState);
//     std::vector<std::uniform_real_distribution<double>> dists;
    
//     for (int j = 0; j < X.cols(); j++) {
//         double minVal = X.col(j).minCoeff();
//         double maxVal = X.col(j).maxCoeff();
//         dists.emplace_back(minVal, maxVal);
//     }
    
//     std::vector<float> uniformVec;
//     uniformVec.reserve(m * X.cols());
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < X.cols(); j++) {
//             uniformVec.push_back(static_cast<float>(dists[j](gen)));
//         }
//     }
    
//     flann::Matrix<float> flannUniform(uniformVec.data(), m, X.cols());
    
//     // 计算实际数据的最近邻距离
//     std::vector<int> indices(n, 0);
//     std::vector<float> distsReal(n, 0.0f);
    
//     try {
//         // 尝试获取2个最近邻
//         std::vector<int> results2(n * 2);
//         std::vector<float> dists2(n * 2);
//         flann::Matrix<int> results2_mat(results2.data(), n, 2);
//         flann::Matrix<float> dists2_mat(dists2.data(), n, 2);
//         index.knnSearch(flannData, results2_mat, dists2_mat, 2, flann::SearchParams(100));
        
//         for (int i = 0; i < n; i++) {
//             distsReal[i] = dists2[i * 2 + 1]; // 第二个最近邻距离
//         }
//     } catch (...) {
//         // 如果失败，获取1个最近邻
//         flann::Matrix<int> indices_mat(indices.data(), n, 1);
//         flann::Matrix<float> distsReal_mat(distsReal.data(), n, 1);
//         index.knnSearch(flannData, indices_mat, distsReal_mat, 1, flann::SearchParams(100));
//     }
    
//     // 计算随机点的最近邻距离
//     std::vector<int> indicesUniform(m, 0);
//     std::vector<float> distsUniform(m, 0.0f);
//     flann::Matrix<int> indicesUniform_mat(indicesUniform.data(), m, 1);
//     flann::Matrix<float> distsUniform_mat(distsUniform.data(), m, 1);
//     index.knnSearch(flannUniform, indicesUniform_mat, distsUniform_mat, 1, flann::SearchParams(100));
    
//     // 计算霍普金斯统计量
//     double sumW = 0.0, sumU = 0.0;
//     for (float d : distsUniform) sumW += d;
//     for (float d : distsReal) sumU += d;
    
//     return sumW / (sumU + sumW);
// }

// // 霍普金斯统计量计算
static double hopkinsStatistic(const ClusterEval::MatrixType& X, double sampleSize, int randomState) {
    int n = X.rows();
    if (n < 5) {
        std::cout << "警告: 样本量过小，霍普金斯统计量可能不可靠" << std::endl;
        return 0.5;
    }
    
    int m = (sampleSize > 0) ? static_cast<int>(sampleSize * n) : std::min(50, n);
    m = std::max(1, m); // 至少取1个样本
    
    // 转换数据为FLANN兼容格式
    std::vector<float> dataVec;
    dataVec.reserve(n * X.cols());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < X.cols(); j++) {
            dataVec.push_back(static_cast<float>(X(i, j)));
        }
    }
    
    // 构建FLANN索引进行最近邻搜索
    flann::Matrix<float> flannData(dataVec.data(), n, X.cols());
    flann::Index<flann::L2<float>> index(flannData, flann::KDTreeSingleIndexParams(10));
    index.buildIndex();
    
    // 生成均匀分布的随机点
    // std::mt19937 gen(randomState);
    std::mt19937 gen(42);
    std::vector<std::uniform_real_distribution<double>> dists;
    
    for (int j = 0; j < X.cols(); j++) {
        double minVal = X.col(j).minCoeff();
        double maxVal = X.col(j).maxCoeff();
        dists.emplace_back(minVal, maxVal);
    }
    
    std::vector<float> uniformVec;
    uniformVec.reserve(m * X.cols());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < X.cols(); j++) {
            uniformVec.push_back(static_cast<float>(dists[j](gen)));
        }
    }
    
    flann::Matrix<float> flannUniform(uniformVec.data(), m, X.cols());
    
    // 计算实际数据的最近邻距离
    std::vector<int> indices(n, 0);
    std::vector<float> distsReal(n, 0.0f);
    
    try {
        // 尝试获取2个最近邻
        std::vector<int> results2(n * 2);
        std::vector<float> dists2(n * 2);
        flann::Matrix<int> results2_mat(results2.data(), n, 2);
        flann::Matrix<float> dists2_mat(dists2.data(), n, 2);
        index.knnSearch(flannData, results2_mat, dists2_mat, 2, flann::SearchParams(100));
        
        for (int i = 0; i < n; i++) {
            distsReal[i] = dists2[i * 2 + 1]; // 第二个最近邻距离
        }
    } catch (...) {
        // 如果失败，获取1个最近邻
        flann::Matrix<int> indices_mat(indices.data(), n, 1);
        flann::Matrix<float> distsReal_mat(distsReal.data(), n, 1);
        index.knnSearch(flannData, indices_mat, distsReal_mat, 1, flann::SearchParams(100));
    }
    
    // 计算随机点的最近邻距离
    std::vector<int> indicesUniform(m, 0);
    std::vector<float> distsUniform(m, 0.0f);
    flann::Matrix<int> indicesUniform_mat(indicesUniform.data(), m, 1);
    flann::Matrix<float> distsUniform_mat(distsUniform.data(), m, 1);
    index.knnSearch(flannUniform, indicesUniform_mat, distsUniform_mat, 1, flann::SearchParams(100));
    
    // 计算霍普金斯统计量
    double sumW = 0.0, sumU = 0.0;
    for (float d : distsUniform) sumW += d;
    for (float d : distsReal) sumU += d;
    
    return sumW / (sumU + sumW);
}



// 分析最近邻距离并计算变异系数
static std::map<std::string, double> analyzeNeighborDistances(const ClusterEval::MatrixType& X, int k) {
    int n = X.rows();
    if (n < 2) {
        std::cout << "警告: 样本量过小，无法计算最近邻距离" << std::endl;
        return {{"cv", 0.0}};
    }
    
    k = std::min(k, n - 1);
    if (k < 1) k = 1; // 至少取1个最近邻
    
    // 转换数据为FLANN兼容格式
    std::vector<float> dataVec;
    dataVec.reserve(n * X.cols());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < X.cols(); j++) {
            dataVec.push_back(static_cast<float>(X(i, j)));
        }
    }
    
    // 构建FLANN索引
    flann::Matrix<float> flannData(dataVec.data(), n, X.cols());
    flann::Index<flann::L2<float>> index(flannData, flann::KDTreeSingleIndexParams(10));
    index.buildIndex();
    
    // 计算k+1个最近邻(包括自身)
    std::vector<int> indices(n * (k + 1), 0);
    std::vector<float> dists(n * (k + 1), 0.0f);
    flann::Matrix<int> indices_mat(indices.data(), n, k + 1);
    flann::Matrix<float> dists_mat(dists.data(), n, k + 1);
    index.knnSearch(flannData, indices_mat, dists_mat, k + 1, flann::SearchParams(100));
    
    // 计算平均最近邻距离(排除自身)
    std::vector<double> avgDistances;
    avgDistances.reserve(n);
    for (int i = 0; i < n; i++) {
        double sumDist = 0.0;
        for (int j = 1; j <= k; j++) { // 从第2个开始(排除自身)
            sumDist += dists[i * (k + 1) + j];
        }
        avgDistances.push_back(sumDist / k);
    }
    
    // 计算统计量
    double mean = 0.0, stddev = 0.0, minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest(), sum = 0.0, sumSq = 0.0;
    
    for (double d : avgDistances) {
        sum += d;
        minVal = std::min(minVal, d);
        maxVal = std::max(maxVal, d);
    }
    
    mean = sum / n;
    
    for (double d : avgDistances) {
        sumSq += (d - mean) * (d - mean);
    }
    
    stddev = std::sqrt(sumSq / n);
    
    // 排序计算中位数
    std::sort(avgDistances.begin(), avgDistances.end());
    double median = (n % 2 == 0) ? 
                   (avgDistances[n/2 - 1] + avgDistances[n/2]) / 2.0 :
                   avgDistances[n/2];
    
    // 计算变异系数
    double cv = (mean > std::numeric_limits<double>::epsilon()) ? stddev / mean : 0.0;
    
    return {
        {"mean", mean},
        {"std", stddev},
        {"min", minVal},
        {"max", maxVal},
        {"median", median},
        {"cv", cv}
    };
}

// 评估聚类适用性
static std::pair<std::map<std::string, std::string>, int> evaluateClusteringSuitability(
    double hopkins, 
    const std::map<std::string, double>& distStats
) {
    // 霍普金斯统计量评估
    std::string hopkinsEval;
    if (hopkins > 0.7) hopkinsEval = "适合聚类";
    else if (hopkins > 0.5) hopkinsEval = "可能适合聚类";
    else hopkinsEval = "不适合聚类";
    
    // 距离变异系数评估
    double distCv = distStats.at("cv");
    std::string distEval;
    if (distCv > 0.5) distEval = "适合距离聚类";
    else if (distCv > 0.2) distEval = "可能适合距离聚类";
    else distEval = "不适合距离聚类";
    
    // 综合评估
    int score = 0;
    if (hopkins > 0.7) score += 1;
    if (distCv > 0.5) score += 1;
    
    std::string overallEval;
    if (score >= 2) overallEval = "适合聚类";
    else if (score >= 1) overallEval = "可能适合聚类";
    else overallEval = "不适合聚类";
    
    // 推荐算法
    std::vector<std::string> algorithmRec;
    if (distCv > 0.5) {
        algorithmRec.push_back("k-means");
        algorithmRec.push_back("层次聚类");
    }
    
    if (algorithmRec.empty()) {
        algorithmRec.push_back("暂不推荐聚类");
    }
    
    return {
        {
            {"霍普金斯评估", hopkinsEval},
            {"距离变异评估", distEval},
            {"综合评估", overallEval},
            {"推荐算法", algorithmRec[0]}
        },
        score
    };
}

// // 评估函数实现
// int evaluateClusteringDataset(const std::vector<Point>& points) {
//     try {
//         // 检查输入数据
//         if (points.empty()) {
//             throw std::runtime_error("输入数据为空");
//         }
        
//         size_t numPoints = points.size();
        
//         std::cout << "\n开始分析数据集: " << numPoints << " 个点" << std::endl;
        
//         // 1. 将Point向量转换为Eigen矩阵
//         auto X = pointsToMatrix(points);
        
//         // 2. 霍普金斯统计量计算
//         double hopkins = hopkinsStatistic(X);
        
//         // 3. 距离变异系数计算
//         auto distStats = analyzeNeighborDistances(X);
        
//         // 4. 聚类适用性评估
//         auto suitabilityPair = evaluateClusteringSuitability(hopkins, distStats);
//         auto suitability = suitabilityPair.first;
//         int score = suitabilityPair.second;
        
//         // 5. 结果展示
//         std::cout << "\n=== 聚类需求评估结果 ===" << std::endl;
//         std::cout << "\n1. 霍普金斯统计量: " << std::fixed << std::setprecision(3) << hopkins 
//                   << " [" << suitability.at("霍普金斯评估") << "]" << std::endl;
//         std::cout << "2. 距离变异系数: " << std::fixed << std::setprecision(3) << distStats.at("cv") 
//                   << " [" << suitability.at("距离变异评估") << "]" << std::endl;
        
//         // 综合评估
//         std::cout << "\n=== 综合评估结论 ===" << std::endl;
//         std::cout << "综合评估: " << suitability.at("综合评估") 
//                   << " (得分: " << score << "/2)" << std::endl;
//         std::cout << "推荐算法: " << suitability.at("推荐算法") << std::endl;
        
//         std::cout << "\n分析完成！" << std::endl;
        
//         return score;
//     } catch (const std::exception& e) {
//         std::cout << "分析过程中出错: " << e.what() << std::endl;
//         throw; // 抛出异常以便外部处理
//     }
// }

// // 评估函数实现
int evaluateClusteringDataset(const std::vector<Point>& points) {
    try {
        // 检查输入数据
        if (points.empty()) {
            throw std::runtime_error("输入数据为空");
        }
        
        size_t numPoints = points.size();
        
        std::cout << "\n开始分析数据集: " << numPoints << " 个点" << std::endl;
        
        // 1. 将Point向量转换为Eigen矩阵
        auto X = pointsToMatrix(points);
        
        // 2. 霍普金斯统计量计算
        double hopkins = hopkinsStatistic(X, 0.3, 42);
        
        // 3. 距离变异系数计算
        auto distStats = analyzeNeighborDistances(X);
        
        // 4. 聚类适用性评估
        auto suitabilityPair = evaluateClusteringSuitability(hopkins, distStats);
        auto suitability = suitabilityPair.first;
        int score = suitabilityPair.second;
        
        // 5. 结果展示
        std::cout << "\n=== 聚类需求评估结果 ===" << std::endl;
        std::cout << "\n1. 霍普金斯统计量: " << std::fixed << std::setprecision(3) << hopkins 
                  << " [" << suitability.at("霍普金斯评估") << "]" << std::endl;
        std::cout << "2. 距离变异系数: " << std::fixed << std::setprecision(3) << distStats.at("cv") 
                  << " [" << suitability.at("距离变异评估") << "]" << std::endl;
        
        // 综合评估
        std::cout << "\n=== 综合评估结论 ===" << std::endl;
        std::cout << "综合评估: " << suitability.at("综合评估") 
                  << " (得分: " << score << "/2)" << std::endl;
        std::cout << "推荐算法: " << suitability.at("推荐算法") << std::endl;
        
        std::cout << "\n分析完成！" << std::endl;
        
        return score;
    } catch (const std::exception& e) {
        std::cout << "分析过程中出错: " << e.what() << std::endl;
        throw; // 抛出异常以便外部处理
    }
}
