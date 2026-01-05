// #include "k-means_clustering.h"
// #include <random>
// #include <limits>
// #include <cmath>
// #include <algorithm>
// #include <iostream>
// #include <chrono>   // 添加时间计算支持

// using namespace std;
// using namespace std::chrono;  // 时间计算命名空间

// // KMeans类的完整实现
// class KMeans {
// public:
//     // 带k值自动确定的聚类
//     KMeansResult auto_cluster(vector<Point>& points, 
//                               int max_k, 
//                               int min_k,
//                               int max_iters,
//                               int trials_per_k) {
//         // 输出标题
//         cout << "\n==========================================";
//         cout << "\nStarting K-means clustering with auto-k selection";
//         cout << "\nPoints count: " << points.size();
//         cout << "\nK range: [" << min_k << " - " << max_k << "]";
//         cout << "\nMax iterations: " << max_iters;
//         cout << "\nTrials per k: " << trials_per_k;
//         cout << "\n------------------------------------------" << endl;
        
//         if (points.empty()) {
//             cerr << "Warning: Empty point set provided!" << endl;
//             return {{}, {}, 0, -1.0};
//         }
        
//         auto start_time = high_resolution_clock::now();  // 整体开始时间
        
//         // 确保合理的k值范围
//         int n = points.size();
//         min_k = max(2, min_k);
//         max_k = min(max_k, n);
        
//         vector<double> scores;
//         vector<KMeansResult> results;
        
//         // 输出参数设置
//         cout << "\nEvaluating cluster sizes from k = " << min_k << " to " << max_k << "...\n";
        
//         // 评估不同k值下的聚类质量
//         for (int k = min_k; k <= max_k; ++k) {
//             cout << "\n------------------------------------------";
//             cout << "\nEvaluating k = " << k;
//             cout << "\n------------------------------------------" << endl;
            
//             vector<KMeansResult> trial_results;
//             vector<double> trial_scores;
            
//             // 多次运行取最优
//             for (int t = 0; t < trials_per_k; ++t) {
//                 auto trial_start = high_resolution_clock::now();  // 单次运行开始时间
                
//                 cout << "\nTrial #" << t+1 << " for k=" << k;
//                 auto result = cluster_with_k(points, k, max_iters);
//                 double score = result.silhouette_score;
                
//                 auto trial_end = high_resolution_clock::now();  // 单次运行结束时间
//                 auto trial_duration = duration_cast<milliseconds>(trial_end - trial_start);
                
//                 cout << " - Silhouette score: " << score
//                      << " - Time: " << trial_duration.count() << " ms" << endl;
                
//                 trial_results.push_back(result);
//                 trial_scores.push_back(score);
//             }
            
//             // 选择本次k值下最好的一次运行
//             auto best_trial_iter = max_element(trial_scores.begin(), trial_scores.end());
//             auto best_trial_idx = distance(trial_scores.begin(), best_trial_iter);
//             double best_score = *best_trial_iter;
            
//             cout << "\nBest trial for k=" << k << ": Silhouette = " << best_score << endl;
            
//             results.push_back(trial_results[best_trial_idx]);
//             scores.push_back(best_score);
//         }
        
//         // 找到最佳k值
//         auto best_k_iter = max_element(scores.begin(), scores.end());
//         auto best_k_idx = distance(scores.begin(), best_k_iter);
//         int best_k = min_k + best_k_idx;
        
//         auto end_time = high_resolution_clock::now();  // 整体结束时间
//         auto total_duration = duration_cast<milliseconds>(end_time - start_time);
        
//         // 输出最终结果
//         cout << "\n------------------------------------------";
//         cout << "\nOptimal cluster count: k = " << best_k;
//         cout << "\nBest silhouette score: " << scores[best_k_idx];
//         cout << "\nTotal time: " << total_duration.count() << " ms";
//         cout << "\n==========================================" << endl;
        
//         return results[best_k_idx];
//     }
    
//     // 指定k值的聚类函数
//     KMeansResult cluster_with_k(vector<Point>& points, int k, int max_iters=100) {
//         auto start_time = high_resolution_clock::now();  // 开始时间
        
//         int n = points.size();
//         KMeansResult result;
//         result.k = k;
        
//         // 输出标题
//         cout << "\n==========================================";
//         cout << "\nStarting K-means clustering with k = " << k;
//         cout << "\nPoints count: " << n;
//         cout << "\nMax iterations: " << max_iters;
//         cout << "\n------------------------------------------" << endl;
        
//         // 处理边界情况
//         if (n == 0 || k <= 0) {
//             cerr << "Invalid parameters: n = " << n << ", k = " << k << endl;
//             result.silhouette_score = -1.0;
//             return result;
//         }
        
//         // 确保k不超过点数
//         if (k > n) {
//             k = n;
//             cout << "Adjusted k to n: k = " << k << endl;
//         }
//         result.k = k;  // 更新实际k值
        
//         // 初始化质心
//         auto init_start = high_resolution_clock::now();
//         auto centroids = initialize_centroids(points, k);
//         auto init_end = high_resolution_clock::now();
//         auto init_duration = duration_cast<milliseconds>(init_end - init_start);
        
//         cout << "Centroids initialized in " << init_duration.count() << " ms" << endl;
        
//         vector<int> labels(n, -1);
//         int iter = 0;
//         bool changed = true;
        
//         // 主迭代循环
//         while (changed && iter < max_iters) {
//             auto iter_start = high_resolution_clock::now();  // 单次迭代开始
            
//             changed = false;
//             int assignments_changed = 0;
            
//             // 分配阶段: 分配每个点到最近的质心
//             for (int i = 0; i < n; ++i) {
//                 int new_label = -1;
//                 float min_dist = numeric_limits<float>::max();
                
//                 for (int j = 0; j < k; ++j) {
//                     float dist = points[i].cal_dist(centroids[j]);
//                     if (dist < min_dist) {
//                         min_dist = dist;
//                         new_label = j;
//                     }
//                 }
                
//                 if (labels[i] != new_label) {
//                     assignments_changed++;
//                     labels[i] = new_label;
//                     changed = true;
//                 }
//             }
            
//             auto assignment_end = high_resolution_clock::now();
//             auto assignment_duration = duration_cast<milliseconds>(assignment_end - iter_start);
            
//             cout << "Iter " << iter << ": Assignments changed: " << assignments_changed
//                  << ", Assignment time: " << assignment_duration.count() << " ms, ";
            
//             if (!changed) {
//                 cout << "Converged!" << endl;
//                 break;  // 收敛
//             }
            
//             // 更新阶段: 重新计算质心
//             vector<int> counts(k, 0);
//             vector<float> sum_x(k, 0.0);
//             vector<float> sum_y(k, 0.0);
//             int empty_clusters = 0;
            
//             for (int i = 0; i < n; ++i) {
//                 int cluster = labels[i];
//                 sum_x[cluster] += points[i].x;
//                 sum_y[cluster] += points[i].y;
//                 counts[cluster]++;
//             }
            
//             // 更新质心并处理空簇
//             for (int j = 0; j < k; ++j) {
//                 if (counts[j] > 0) {
//                     centroids[j] = Point(sum_x[j] / counts[j], 
//                                          sum_y[j] / counts[j]);
//                 } else {
//                     empty_clusters++;
//                     // 空簇处理
//                     uniform_int_distribution<int> dist(0, n-1);
//                     centroids[j] = points[dist(gen)];
                    
//                     // 输出空簇警告
//                     cout << "\n  WARNING: Cluster " << j << " is empty, reassigning centroid!" << endl;
//                 }
//             }
            
//             auto update_end = high_resolution_clock::now();
//             auto update_duration = duration_cast<milliseconds>(update_end - assignment_end);
//             auto iter_duration = duration_cast<milliseconds>(update_end - iter_start);
            
//             cout << "Update time: " << update_duration.count() << " ms, "
//                  << "Iter total: " << iter_duration.count() << " ms";
            
//             if (empty_clusters > 0) {
//                 cout << ", Empty clusters: " << empty_clusters;
//             }
//             cout << endl;
            
//             iter++;
//         }
        
//         // 构建簇映射
//         result.clustered_points.clear();
//         for (int i = 0; i < k; ++i) {
//             result.clustered_points[i] = vector<Point>();
//         }
        
//         for (int i = 0; i < n; ++i) {
//             int cluster_id = labels[i];
//             result.clustered_points[cluster_id].push_back(points[i]);
//         }
        
//         result.centroids = centroids;
        
//         // 计算轮廓系数
//         auto silhouette_start = high_resolution_clock::now();
//         result.silhouette_score = calculate_silhouette_score(result);
//         auto silhouette_end = high_resolution_clock::now();
//         auto silhouette_duration = duration_cast<milliseconds>(silhouette_end - silhouette_start);
        
//         cout << "Silhouette calculation: " << silhouette_duration.count() << " ms" << endl;
        
//         auto end_time = high_resolution_clock::now();  // 结束时间
//         auto total_duration = duration_cast<milliseconds>(end_time - start_time);
        
//         // 输出聚类统计信息
//         cout << "\nClustering Summary:";
//         cout << "\n  Total points: " << n;
//         cout << "\n  Cluster count: " << k;
//         cout << "\n  Iterations: " << iter;
//         cout << "\n  Silhouette score: " << result.silhouette_score;
//         cout << "\n  Total time: " << total_duration.count() << " ms";
        
//         // 输出簇大小分布
//         cout << "\nCluster sizes:";
//         for (int i = 0; i < k; ++i) {
//             int size = result.clustered_points[i].size();
//             cout << " " << i << ":" << size;
//         }
//         cout << "\n==========================================" << endl;
        
//         return result;
//     }
    
//     // 计算轮廓系数
//     double calculate_silhouette_score(KMeansResult& result) {
//         if (result.k <= 1 || result.clustered_points.size() <= 1) {
//             return -1.0; // 无效情况
//         }
        
//         double total_score = 0.0;
//         int count = 0;
        
//         cout << "Starting silhouette calculation for " 
//              << result.clustered_points.size() << " clusters..." << endl;
        
//         // 遍历所有点
//         for (auto& cluster : result.clustered_points) {
//             int cluster_id = cluster.first;
//             auto& points_in_cluster = cluster.second;
//             int cluster_size = points_in_cluster.size();
            
//             if (cluster_size == 0) continue;
            
//             cout << "Processing cluster " << cluster_id << " (" 
//                  << cluster_size << " points)..." << endl;
            
//             for (auto& p : points_in_cluster) {
//                 // 计算同簇内平均距离(a)
//                 double a_value = 0.0;
//                 int same_cluster_count = 0;
                
//                 for (auto& other : points_in_cluster) {
//                     if (&p == &other) continue;
//                     a_value += p.cal_dist(other);
//                     same_cluster_count++;
//                 }
                
//                 a_value /= same_cluster_count;
                
//                 // 计算最近邻簇平均距离(b)
//                 double b_value = numeric_limits<double>::max();
//                 int nearest_cluster = -1;
                
//                 for (auto& other_cluster : result.clustered_points) {
//                     int other_id = other_cluster.first;
//                     if (other_id == cluster_id) continue;
                    
//                     auto& other_points = other_cluster.second;
//                     if (other_points.empty()) continue;
                    
//                     double dist_sum = 0.0;
//                     for (auto& op : other_points) {
//                         dist_sum += p.cal_dist(op);
//                     }
//                     double avg_dist = dist_sum / other_points.size();
                    
//                     if (avg_dist < b_value) {
//                         b_value = avg_dist;
//                         nearest_cluster = other_id;
//                     }
//                 }
                
//                 // 处理未找到最近簇的情况
//                 if (b_value == numeric_limits<double>::max()) {
//                     b_value = a_value; // 没有其他簇，使用a值代替
//                     nearest_cluster = cluster_id;
//                 }
                
//                 // 计算该点的轮廓系数
//                 double s = 0.0;
//                 if (fabs(a_value - b_value) < 1e-9) {
//                     s = 0.0; // 防止除以零
//                 } else if (a_value < b_value) {
//                     s = 1.0 - a_value / b_value;
//                 } else if (a_value > b_value) {
//                     s = b_value / a_value - 1.0;
//                 }
                
//                 // 调试输出点轮廓系数
//                 // cout << "Point (" << p.x << "," << p.y << "): a=" << a_value 
//                 //      << ", b=" << b_value << "(n-cluster " << nearest_cluster
//                 //      << "), s=" << s << endl;
                
//                 total_score += s;
//                 count++;
//             }
//         }
        
//         if (count == 0) return -1.0;
        
//         double avg_score = total_score / count;
//         cout << "Silhouette calculation completed. Average: " << avg_score << endl;
//         return avg_score;
//     }

// private:
//     random_device rd;
//     mt19937 gen{rd()};
    
//     // K-means++ 初始化
//     vector<Point> initialize_centroids(vector<Point>& points, int k) {
//         auto start_time = high_resolution_clock::now();  // 初始化开始时间
        
//         vector<Point> centroids;
//         int n = points.size();
        
//         if (k == 0 || n == 0) return centroids;
        
//         cout << "Starting K-means++ initialization for " << k << " centroids..." << endl;
        
//         // 随机选择第一个质心
//         uniform_int_distribution<int> dist(0, n-1);
//         int first_idx = dist(gen);
//         centroids.push_back(points[first_idx]);
        
//         cout << "  Centroid 0: Point #" << first_idx 
//              << " (" << points[first_idx].x << "," << points[first_idx].y << ")" << endl;
        
//         // 选择剩余k-1个质心
//         for (int i = 1; i < k; ++i) {
//             vector<float> min_dists(n, numeric_limits<float>::max());
//             float total_sq_dist = 0.0;
            
//             // 计算每个点到最近质心的距离
//             for (int j = 0; j < n; ++j) {
//                 float min_dist = numeric_limits<float>::max();
//                 for (auto& centroid : centroids) {
//                     Point& non_const_cent = const_cast<Point&>(centroid);
//                     float dist = points[j].cal_dist(non_const_cent);
//                     if (dist < min_dist) min_dist = dist;
//                 }
//                 min_dists[j] = min_dist;
//                 total_sq_dist += min_dist * min_dist;
//             }
            
//             // 概率选择下一个质心
//             uniform_real_distribution<float> prob_dist(0.0, total_sq_dist);
//             float threshold = prob_dist(gen);
//             float cumulative = 0.0;
//             int selected_idx = -1;
            
//             for (int j = 0; j < n; ++j) {
//                 cumulative += min_dists[j] * min_dists[j];
//                 if (cumulative >= threshold) {
//                     selected_idx = j;
//                     break;
//                 }
//             }
            
//             if (selected_idx == -1) {
//                 selected_idx = n-1;  // fallback
//             }
            
//             centroids.push_back(points[selected_idx]);
            
//             // 输出选择的质心
//             cout << "  Centroid " << i << ": Point #" << selected_idx 
//                  << " (" << points[selected_idx].x << "," << points[selected_idx].y 
//                  << "), Distance Weight: " << min_dists[selected_idx] << endl;
//         }
        
//         auto end_time = high_resolution_clock::now();  // 初始化结束时间
//         auto duration = duration_cast<milliseconds>(end_time - start_time);
        
//         cout << "K-means++ initialization completed in " 
//              << duration.count() << " ms" << endl;
        
//         return centroids;
//     }
// };

// // 主接口函数的具体实现
// KMeansResult perform_kmeans_clustering(
//     vector<Point>& points, 
//     int k,
//     int max_k,
//     int min_k,
//     int max_iters,
//     int trials_per_k)
// {
//     KMeans kmeans;
    
//     // 记录总时间
//     auto start_time = high_resolution_clock::now();
    
//     if (k <= 0) {
//         cout << "\n\n**************************************************";
//         cout << "\n* K-MEANS CLUSTERING WITH AUTO-K SELECTION STARTED *";
//         cout << "\n**************************************************";
//         // 自动确定最佳k值
//         return kmeans.auto_cluster(points, max_k, min_k, max_iters, trials_per_k);
//     } else {
//         cout << "\n\n************************************************";
//         cout << "\n*   K-MEANS CLUSTERING WITH FIXED K = " << k << " STARTED   *";
//         cout << "\n************************************************";
//         // 使用指定k值
//         return kmeans.cluster_with_k(points, k, max_iters);
//     }
// }


#include "k-means_clustering.h"
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>   // 添加时间计算支持

using namespace std;
using namespace std::chrono;  // 时间计算命名空间
std::mt19937 gen{42};
// KMeans类的完整实现
class KMeans {
public:
    // 带k值自动确定的聚类
    KMeansResult auto_cluster(vector<Point>& points, 
                              int max_k, 
                              int min_k,
                              int max_iters,
                              int trials_per_k) {
        // 输出标题
        cout << "\n==========================================";
        cout << "\nStarting K-means clustering with auto-k selection";
        cout << "\nPoints count: " << points.size();
        cout << "\nK range: [" << min_k << " - " << max_k << "]";
        cout << "\nMax iterations: " << max_iters;
        cout << "\nTrials per k: " << trials_per_k;
        cout << "\n------------------------------------------" << endl;
        
        if (points.empty()) {
            cerr << "Warning: Empty point set provided!" << endl;
            return {{}, {}, 0, -1.0};
        }
        
        auto start_time = high_resolution_clock::now();  // 整体开始时间
        
        // 确保合理的k值范围
        int n = points.size();
        min_k = max(2, min_k);
        max_k = min(max_k, n);
        
        vector<double> scores;
        vector<KMeansResult> results;
        
        // 输出参数设置
        cout << "\nEvaluating cluster sizes from k = " << min_k << " to " << max_k << "...\n";
        
        // 评估不同k值下的聚类质量
        for (int k = min_k; k <= max_k; ++k) {
            cout << "\n------------------------------------------";
            cout << "\nEvaluating k = " << k;
            cout << "\n------------------------------------------" << endl;
            
            vector<KMeansResult> trial_results;
            vector<double> trial_scores;
            
            // 多次运行取最优
            for (int t = 0; t < trials_per_k; ++t) {
                auto trial_start = high_resolution_clock::now();  // 单次运行开始时间
                
                cout << "\nTrial #" << t+1 << " for k=" << k;
                auto result = cluster_with_k(points, k, max_iters);
                double score = result.silhouette_score;
                
                auto trial_end = high_resolution_clock::now();  // 单次运行结束时间
                auto trial_duration = duration_cast<milliseconds>(trial_end - trial_start);
                
                cout << " - Silhouette score: " << score
                     << " - Time: " << trial_duration.count() << " ms" << endl;
                
                trial_results.push_back(result);
                trial_scores.push_back(score);
            }
            
            // 选择本次k值下最好的一次运行
            auto best_trial_iter = max_element(trial_scores.begin(), trial_scores.end());
            auto best_trial_idx = distance(trial_scores.begin(), best_trial_iter);
            double best_score = *best_trial_iter;
            
            cout << "\nBest trial for k=" << k << ": Silhouette = " << best_score << endl;
            
            results.push_back(trial_results[best_trial_idx]);
            scores.push_back(best_score);
        }
        
        // 找到最佳k值
        auto best_k_iter = max_element(scores.begin(), scores.end());
        auto best_k_idx = distance(scores.begin(), best_k_iter);
        int best_k = min_k + best_k_idx;
        
        auto end_time = high_resolution_clock::now();  // 整体结束时间
        auto total_duration = duration_cast<milliseconds>(end_time - start_time);
        
        // 输出最终结果
        cout << "\n------------------------------------------";
        cout << "\nOptimal cluster count: k = " << best_k;
        cout << "\nBest silhouette score: " << scores[best_k_idx];
        cout << "\nTotal time: " << total_duration.count() << " ms";
        cout << "\n==========================================" << endl;
        
        return results[best_k_idx];
    }
    
    // 指定k值的聚类函数
    KMeansResult cluster_with_k(vector<Point>& points, int k, int max_iters=100) {
        auto start_time = high_resolution_clock::now();  // 开始时间
        gen.seed(42); // 重置全局gen的种子
        int n = points.size();
        KMeansResult result;
        result.k = k;
        
        // 输出标题
        cout << "\n==========================================";
        cout << "\nStarting K-means clustering with k = " << k;
        cout << "\nPoints count: " << n;
        cout << "\nMax iterations: " << max_iters;
        cout << "\n------------------------------------------" << endl;
        
        // 处理边界情况
        if (n == 0 || k <= 0) {
            cerr << "Invalid parameters: n = " << n << ", k = " << k << endl;
            result.silhouette_score = -1.0;
            return result;
        }
        
        // 确保k不超过点数
        if (k > n) {
            k = n;
            cout << "Adjusted k to n: k = " << k << endl;
        }
        result.k = k;  // 更新实际k值
        
        // 初始化质心
        auto init_start = high_resolution_clock::now();
        auto centroids = initialize_centroids(points, k);
        auto init_end = high_resolution_clock::now();
        auto init_duration = duration_cast<milliseconds>(init_end - init_start);
        
        cout << "Centroids initialized in " << init_duration.count() << " ms" << endl;
        
        vector<int> labels(n, -1);
        int iter = 0;
        bool changed = true;
        
        // 主迭代循环
        while (changed && iter < max_iters) {
            auto iter_start = high_resolution_clock::now();  // 单次迭代开始
            
            changed = false;
            int assignments_changed = 0;
            
            // 分配阶段: 分配每个点到最近的质心
            for (int i = 0; i < n; ++i) {
                int new_label = -1;
                float min_dist = numeric_limits<float>::max();
                
                for (int j = 0; j < k; ++j) {
                    float dist = points[i].cal_dist(centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        new_label = j;
                    }
                }
                
                if (labels[i] != new_label) {
                    assignments_changed++;
                    labels[i] = new_label;
                    changed = true;
                }
            }
            
            auto assignment_end = high_resolution_clock::now();
            auto assignment_duration = duration_cast<milliseconds>(assignment_end - iter_start);
            
            cout << "Iter " << iter << ": Assignments changed: " << assignments_changed
                 << ", Assignment time: " << assignment_duration.count() << " ms, ";
            
            if (!changed) {
                cout << "Converged!" << endl;
                break;  // 收敛
            }
            
            // 更新阶段: 重新计算质心
            vector<int> counts(k, 0);
            vector<vector<float>> sum_coords(k, vector<float>(points[0].dim, 0.0f));
            int empty_clusters = 0;
            
            for (int i = 0; i < n; ++i) {
                int cluster = labels[i];
                for (int d = 0; d < points[0].dim; ++d) {
                    sum_coords[cluster][d] += points[i].coords[d];
                }
                counts[cluster]++;
            }
            
            // 更新质心并处理空簇
            for (int j = 0; j < k; ++j) {
                if (counts[j] > 0) {
                    for (int d = 0; d < points[0].dim; ++d) {
                        centroids[j].coords[d] = sum_coords[j][d] / counts[j];
                    }
                } else {
                    empty_clusters++;
                    // 空簇处理
                    uniform_int_distribution<int> dist(0, n-1);
                    centroids[j] = points[dist(gen)];
                    
                    // 输出空簇警告
                    cout << "\n  WARNING: Cluster " << j << " is empty, reassigning centroid!" << endl;
                }
            }
            
            auto update_end = high_resolution_clock::now();
            auto update_duration = duration_cast<milliseconds>(update_end - assignment_end);
            auto iter_duration = duration_cast<milliseconds>(update_end - iter_start);
            
            cout << "Update time: " << update_duration.count() << " ms, "
                 << "Iter total: " << iter_duration.count() << " ms";
            
            if (empty_clusters > 0) {
                cout << ", Empty clusters: " << empty_clusters;
            }
            cout << endl;
            
            iter++;
        }
        
        // 构建簇映射
        result.clustered_points.clear();
        for (int i = 0; i < k; ++i) {
            result.clustered_points[i] = vector<Point>();
        }
        
        for (int i = 0; i < n; ++i) {
            int cluster_id = labels[i];
            result.clustered_points[cluster_id].push_back(points[i]);
        }
        
        result.centroids = centroids;
        
        // 计算轮廓系数
        auto silhouette_start = high_resolution_clock::now();
        result.silhouette_score = calculate_silhouette_score(result);
        auto silhouette_end = high_resolution_clock::now();
        auto silhouette_duration = duration_cast<milliseconds>(silhouette_end - silhouette_start);
        
        cout << "Silhouette calculation: " << silhouette_duration.count() << " ms" << endl;
        
        auto end_time = high_resolution_clock::now();  // 结束时间
        auto total_duration = duration_cast<milliseconds>(end_time - start_time);
        
        // 输出聚类统计信息
        cout << "\nClustering Summary:";
        cout << "\n  Total points: " << n;
        cout << "\n  Cluster count: " << k;
        cout << "\n  Iterations: " << iter;
        cout << "\n  Silhouette score: " << result.silhouette_score;
        cout << "\n  Total time: " << total_duration.count() << " ms";
        
        // 输出簇大小分布
        cout << "\nCluster sizes:";
        for (int i = 0; i < k; ++i) {
            int size = result.clustered_points[i].size();
            cout << " " << i << ":" << size;
        }
        cout << "\n==========================================" << endl;
        
        return result;
    }
    
    // 计算轮廓系数
    double calculate_silhouette_score(KMeansResult& result) {
        if (result.k <= 1 || result.clustered_points.size() <= 1) {
            return -1.0; // 无效情况
        }
        
        double total_score = 0.0;
        int count = 0;
        
        cout << "Starting silhouette calculation for " 
             << result.clustered_points.size() << " clusters..." << endl;
        
        // 遍历所有点
        for (auto& cluster : result.clustered_points) {
            int cluster_id = cluster.first;
            auto& points_in_cluster = cluster.second;
            int cluster_size = points_in_cluster.size();
            
            if (cluster_size == 0) continue;
            
            cout << "Processing cluster " << cluster_id << " (" 
                 << cluster_size << " points)..." << endl;
            
            for (auto& p : points_in_cluster) {
                // 计算同簇内平均距离(a)
                double a_value = 0.0;
                int same_cluster_count = 0;
                
                for (auto& other : points_in_cluster) {
                    if (&p == &other) continue;
                    a_value += p.cal_dist(other);
                    same_cluster_count++;
                }
                
                a_value /= same_cluster_count;
                
                // 计算最近邻簇平均距离(b)
                double b_value = numeric_limits<double>::max();
                int nearest_cluster = -1;
                
                for (auto& other_cluster : result.clustered_points) {
                    int other_id = other_cluster.first;
                    if (other_id == cluster_id) continue;
                    
                    auto& other_points = other_cluster.second;
                    if (other_points.empty()) continue;
                    
                    double dist_sum = 0.0;
                    for (auto& op : other_points) {
                        dist_sum += p.cal_dist(op);
                    }
                    double avg_dist = dist_sum / other_points.size();
                    
                    if (avg_dist < b_value) {
                        b_value = avg_dist;
                        nearest_cluster = other_id;
                    }
                }
                
                // 处理未找到最近簇的情况
                if (b_value == numeric_limits<double>::max()) {
                    b_value = a_value; // 没有其他簇，使用a值代替
                    nearest_cluster = cluster_id;
                }
                
                // 计算该点的轮廓系数
                double s = 0.0;
                if (fabs(a_value - b_value) < 1e-9) {
                    s = 0.0; // 防止除以零
                } else if (a_value < b_value) {
                    s = 1.0 - a_value / b_value;
                } else if (a_value > b_value) {
                    s = b_value / a_value - 1.0;
                }
                
                // 调试输出点轮廓系数
                total_score += s;
                count++;
            }
        }
        
        if (count == 0) return -1.0;
        
        double avg_score = total_score / count;
        cout << "Silhouette calculation completed. Average: " << avg_score << endl;
        return avg_score;
    }

private:
    // random_device rd;
    // mt19937 gen{rd()};
    // mt19937 gen{42};

    // K-means++ 初始化
    vector<Point> initialize_centroids(vector<Point>& points, int k) {
        auto start_time = high_resolution_clock::now();  // 初始化开始时间
        
        vector<Point> centroids;
        int n = points.size();
        
        if (k == 0 || n == 0) return centroids;
        
        cout << "Starting K-means++ initialization for " << k << " centroids..." << endl;
        
        // 随机选择第一个质心
        uniform_int_distribution<int> dist(0, n-1);
        int first_idx = dist(gen);
        centroids.push_back(points[first_idx]);
        
        cout << "  Centroid 0: Point #" << first_idx 
             << " (" << points[first_idx].coords[0] << ", " << points[first_idx].coords[1] << ")" << endl;
        
        // 选择剩余k-1个质心
        for (int i = 1; i < k; ++i) {
            vector<float> min_dists(n, numeric_limits<float>::max());
            float total_sq_dist = 0.0;
            
            // 计算每个点到最近质心的距离
            for (int j = 0; j < n; ++j) {
                float min_dist = numeric_limits<float>::max();
                for (auto& centroid : centroids) {
                    float dist = points[j].cal_dist(centroid);
                    if (dist < min_dist) min_dist = dist;
                }
                min_dists[j] = min_dist;
                total_sq_dist += min_dist * min_dist;
            }
            
            // 概率选择下一个质心
            uniform_real_distribution<float> prob_dist(0.0, total_sq_dist);
            float threshold = prob_dist(gen);
            float cumulative = 0.0;
            int selected_idx = -1;
            
            for (int j = 0; j < n; ++j) {
                cumulative += min_dists[j] * min_dists[j];
                if (cumulative >= threshold) {
                    selected_idx = j;
                    break;
                }
            }
            
            if (selected_idx == -1) {
                selected_idx = n-1;  // fallback
            }
            
            centroids.push_back(points[selected_idx]);
            
            // 输出选择的质心
            cout << "  Centroid " << i << ": Point #" << selected_idx 
                 << " (" << points[selected_idx].coords[0] << ", " << points[selected_idx].coords[1] << ")" << endl;
        }
        
        auto end_time = high_resolution_clock::now();  // 初始化结束时间
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        cout << "K-means++ initialization completed in " 
             << duration.count() << " ms" << endl;
        
        return centroids;
    }
};

// 主接口函数的具体实现
KMeansResult perform_kmeans_clustering(
    vector<Point>& points, 
    int k,
    int max_k,
    int min_k,
    int max_iters,
    int trials_per_k)
{
    KMeans kmeans;
    // k = 2;
    // 记录总时间
    auto start_time = high_resolution_clock::now();
    
    if (k <= 0) {
        cout << "\n\n**************************************************";
        cout << "\n* K-MEANS CLUSTERING WITH AUTO-K SELECTION STARTED *";
        cout << "\n**************************************************";
        // 自动确定最佳k值
        return kmeans.auto_cluster(points, max_k, min_k, max_iters, trials_per_k);
    } else {
        cout << "\n\n************************************************";
        cout << "\n*   K-MEANS CLUSTERING WITH FIXED K = " << k << " STARTED   *";
        cout << "\n************************************************";
        // 使用指定k值
        return kmeans.cluster_with_k(points, k, max_iters);
    }
}