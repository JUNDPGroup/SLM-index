#include <iostream>
#include <string.h>
#include <unordered_set>
#include "ExpRecorder.h"
#include "Constants.h"
using namespace std;
extern Constants constants;
ExpRecorder::ExpRecorder()
{
}

string ExpRecorder::get_time()
{
    return "time:" + to_string(time) + "\n";
}

string ExpRecorder::get_time_size_errors()
{
    string result = "time:" + to_string(time) + "\n" + "train_time:" + to_string(train_time) + "\n" + "encrypt_time:" + to_string(encrypt_time) + "\n" + "cluster_time:" + to_string(cluster_time) + "\n" + "size:" + to_string(size) + "\n" + "maxError:" + to_string(max_error) + "\n" + "min_error:" + to_string(min_error) + "\n" + "leaf_node_num:" + to_string(leaf_node_num) + "\n" + "RSMI_leaf_node_num:" + to_string(RSMI_leaf_node_num) +"\n" + "no_leaf_node_num:" + to_string(non_leaf_node_num) + "\n" + "res_net_num:" + to_string(res_net_num)+ "\n" + "class_net_num:" + to_string(class_net_num) + "\n" + "average_max_error:" + to_string(average_max_error) + "\n" + "average_min_error:" + to_string(average_min_error) + "\n" + "depth:" + to_string(depth) + "\n";
    time = 0;
    train_time = 0;
    encrypt_time = 0;
    cluster_time = 0;
    size = 0;
    max_error = 0;
    min_error = 0;
    average_min_error = 0;
    average_max_error = 0;
    res_net_num = 0;
    class_net_num = 0;
    non_leaf_node_num = 0;
    leaf_node_num = 0;
    RSMI_leaf_node_num = 0;
    depth = 0;
    return result;
}

string ExpRecorder::get_time_size()
{
    string result = "time:" + to_string(time) + "\n" + "size:" + to_string(size) + "\n";
    time = 0;
    size = 0;
    return result;
}

string ExpRecorder::get_time_accuracy()
{
    string result = "time:" + to_string(time) + "\n" + "accuracy:" + to_string(accuracy) + "\n";
    time = 0;
    accuracy = 0;
    return result;
}

string ExpRecorder::get_time_pageaccess_accuracy()
{
    string result = "selectivity: " + to_string(selectivity) + "\n" + "time:" + to_string(time) + "\n" + "min_tim: " + to_string(min_time) + "\n" + "fb_time:" + to_string(fb_time) + "\n" + "noise_time:" + to_string(noise_time) + "\n" + "compare_time:" + to_string(compare_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n" + "accuracy:" + to_string(accuracy) + "\n";
    time = 0;
    page_access = 0;
    accuracy = 0;
    min_time = 0;
    fb_time = 0;
    noise_time = 0;
    compare_time = 0;
    return result;
}

string ExpRecorder::get_enc_depth_time_pageaccess_accuracy()
{
    string result = "selectivity: " + to_string(selectivity) + "\n" + "enc_depth:" + to_string(enc_depth) +"\n" + "time:" + to_string(time) + "\n" + "min_tim: " + to_string(min_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n" + "accuracy:" + to_string(accuracy) + "\n";
    time = 0;
    page_access = 0;
    accuracy = 0;
    min_time = 0;
    enc_depth = 0;
    return result;
}


// string ExpRecorder::get_time_pageaccess()
// {
//     string result = "time:" + to_string(time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
//     time = 0;
//     page_access = 0;
//     return result;
// }
string ExpRecorder::get_time_pageaccess()
{
    string result="time_result:\n";
    for(int i=0;i<time_list.size();i++){
        result.append(to_string(time_list[i])+"\n" );
    }
    result += "mintime:" + to_string(time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_delete_time_pageaccess()
{
    string result = "time:" + to_string(delete_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_insert_time_pageaccess()
{
    string result = "time:" + to_string(insert_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_insert_time_pageaccess_rebuild()
{
    string result = "time:" + to_string(insert_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n" + "rebuild_num:" + to_string(rebuild_num) + "\n" + "rebuild_time:" + to_string(rebuild_time) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_size()
{
    string result = "size:" + to_string(size) + "\n";
    size = 0;
    return result;
}

void ExpRecorder::cal_size()
{
    size = (Constants::DIM * constants.PAGESIZE * Constants::EACH_DIM_LENGTH + constants.PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * leaf_node_num + non_leaf_node_num * Constants::EACH_DIM_LENGTH;
    // size = (Constants::DIM * Constants::PAGESIZE * Constants::EACH_DIM_LENGTH + Constants::PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * leaf_node_num + non_leaf_node_num * Constants::EACH_DIM_LENGTH;
}

void ExpRecorder::clean()
{
    index_high = 0;
    index_low = 0;

    net_class_num = 0;
    leaf_node_num = 0;
    RSMI_leaf_node_num = 0;
    non_leaf_node_num = 0;
    class_net_num = 0;
    res_net_num = 0;

    window_query_result_size = 0;
    acc_window_query_result_size = 0;

    Base_line_window_query_result_size = 0;
    Secure_window_query_result_size = 0;

    pointSet.clear();
    // pointSet.shrink_to_fit();

    knn_query_results.clear();
    knn_query_results.shrink_to_fit();

    Base_line_knn_query_results.clear();
    Base_line_knn_query_results.shrink_to_fit();

    Secure_knn_query_results.clear();
    Secure_knn_query_results.shrink_to_fit();

    acc_knn_query_results.clear();
    acc_knn_query_results.shrink_to_fit();

    time = 0;
    page_access = 0;
    accuracy = 0;
    size = 0;

    result = 0;

    window_query_results.clear();
    window_query_results.shrink_to_fit();

    Base_line_window_query_results.clear();
    Base_line_window_query_results.shrink_to_fit();

    Secure_window_query_results.clear();
    Secure_window_query_results.shrink_to_fit();

    train_time = 0;
    encrypt_time = 0;
    fb_time = 0;
    compare_time = 0;
    noise_time = 0;
    rebuild_num = 0;
    rebuild_time = 0;
    max_error = 0;
    min_error = 0;

    average_min_error = 0;
    average_max_error = 0;

    last_level_model_num = 0;
    // depth = 0;
    enc_depth = 0;
}
