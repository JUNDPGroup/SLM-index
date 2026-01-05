#ifndef EXPRECORDER_H
#define EXPRECORDER_H

#include <vector>
#include "../entities/Point.h"
#include <string>
#include <optional>
#include "Constants.h"
#include "SortTools.h"
#include <queue>
#include <unordered_set>
// #include "../Paillier/paillier.h"
#include <ophelib/paillier_fast.h>
using namespace ophelib;
using namespace std;

extern PaillierFast paillier;

class ExpRecorder
{
    Constants constants;
public:

    priority_queue<Point , vector<Point>, sortForKNN2> pq;

    priority_queue<Point , vector<Point>, enc_sortForKNN2> enc_pq;

    std::unordered_set<Point, PointHash> pointSet;

    long long index_high;
    long long index_low;

    long long leaf_node_num = 0;
    long long non_leaf_node_num = 0;
    long long RSMI_leaf_node_num = 0;
    long long class_net_num = 0;
    long long res_net_num = 0;

    int max_error = 0;
    int min_error = 0;

    int depth = 0;

    int enc_depth = 0;

    // 序列化 depth 到二进制流
    void serializeDepth(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    }

    // 从二进制流反序列化 depth
    void deserializeDepth(std::istream& is) {
        is.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    }


    int result = 0;

    int dim = 0;

    int net_class_num = 0;
    // int wl = 0, rl = 0;

    float selectivity = 0;

    long long total_depth;

    int N = constants.THRESHOLD;
    int B = constants.PAGESIZE;
    int K = constants.Key;

    long long average_max_error = 0;
    long long average_min_error = 0;

    int last_level_model_num = 0;

    // int class_net_num = 0;
    // int RSMI_no_leaf_node_num = 0;
    // int RSMI_leaf_node_num = 0;

    string structure_name;
    string distribution;
    long dataset_cardinality;

    long long insert_num;
    long delete_num;
    float window_size;
    float window_ratio;
    int k_num;
    int skewness = 1;

    long time = 0;
    long min_time = 0;
    long train_time = 0;
    long encrypt_time = 0;
    long cluster_time = 0;
    long fb_time = 0;
    long compare_time = 0;
    long noise_time = 0;

    long insert_time;
    long delete_time;
    long long rebuild_time;
    int rebuild_num;
    double page_access = 1.0;
    double accuracy;
    long size;

    int window_query_result_size;
    int acc_window_query_result_size;
    int Base_line_window_query_result_size;
    int Secure_window_query_result_size;

    vector<Point> knn_query_results;
    vector<Point> acc_knn_query_results;

    vector<Point> window_query_results;

    vector<Point> Base_line_knn_query_results;
    // vector<Point> Base_line_acc_knn_query_results;

    vector<Point> Base_line_window_query_results;

    vector<Point> Secure_window_query_results;

    vector<Point> Secure_knn_query_results;
    vector<long> time_list;

    ExpRecorder();
    string get_time();
    string get_time_pageaccess();
    string get_time_accuracy();
    string get_time_pageaccess_accuracy();
    string get_enc_depth_time_pageaccess_accuracy();
    string get_insert_time_pageaccess_rebuild();
    string get_size();
    string get_time_size();
    string get_time_size_errors();

    string get_insert_time_pageaccess();
    string get_delete_time_pageaccess();
    void cal_size();
    void clean();
};

#endif