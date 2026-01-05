#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include "utils/FileReader.h"
// #include "indices/ZM.h"
#include "indices/RSMI.h"
#include "entities/LeafNode.h"
#include "entities/Node.h"
#include "entities/NonLeafNode.h"
#include "entities/Point.h"
#include "entities/Mbr.h"
#include "utils/ExpRecorder.h"
#include "utils/Constants.h"
#include "utils/FileWriter.h"
#include "utils/util.h"
#include "utils/Modelools_classifierNet.h"
#include <torch/torch.h>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <unordered_set>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>


#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

// #include "Paillier/paillier.h"
#include <ophelib/paillier_fast.h>

#include "agreements/SIC.h"
#include "agreements/SM.h"
#include "agreements/SSED.h"
#include <gmpxx.h>
#include <gmp.h>
#include "agreements/SQ.h"
#include "Serialize/serialization_helpers.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>  // 序列化 std::shared_ptr
#include <boost/serialization/unique_ptr.hpp>  // 序列化 std::unique_ptr
#include <boost/serialization/weak_ptr.hpp>    // 序列化 std::weak_ptr
#include <random>
#include <algorithm>
using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace ophelib;
using namespace std;

#ifndef use_gpu
// #define use_gpu
    // extern Paillier paillier;

    
    float areas[] = {0.0025, 0.005, 0.01, 0.02, 0.04};
    float ratios[] = {0.25, 0.5, 1, 2, 4};
    int Ns[] = {5000, 2500, 500};
    int n_length = sizeof(Ns) / sizeof(Ns[0]);


    int query_window_num = 10;

    int wl = 0, rl = 2;
    int sl = 0, el = 0;

    float selectivity[] = {0.0025, 0.005, 0.01, 0.02, 0.04};
    float enc[] = {0.2, 0.4, 0.6, 0.8, 1.0};


    int query_k_num = 10;


    long long cardinality = 10000;

    long long inserted_num = cardinality / 10;
    string distribution = Constants::DEFAULT_DISTRIBUTION;
    int inserted_partition = 5;
    int skewness = 1;

    PaillierFast paillier(1024);
    // PaillierFast paillier;

    double knn_diff(vector<Point> acc, vector<Point> pred)
    {
        int num = 0;
        for (Point point : pred)
        {
            for (Point point1 : acc)
            {
                int flag = 0;
                for(int i = 0; i < point.dim; i++)
                {
                    if(point.coords[i] != point1.coords[i]) flag++;
                }
                if(!flag) 
                {
                    num++;
                    break;
                    // continue;
                }
            }
        }
        cout<<"num: "<<num<<endl;
        cout<<"size: "<<acc.size()<<endl;
        cout<<"size: "<<pred.size()<<endl;
        cout<<"accuracy: "<<num * 1.0 / pred.size()<<endl;
        return num * 1.0 / pred.size();
    }

    void serialize_depth(const ExpRecorder& recorder, const string& filename) {
        ofstream ofs(filename, ios::binary);
        if (!ofs) {
            cerr << "Error: Failed to open file for writing: " << filename << endl;
            return;
        }
        recorder.serializeDepth(ofs);
        ofs.close();
        cout << "Depth serialized to: " << filename << endl;
    }

    void deserialize_depth(ExpRecorder& recorder, const string& filename) {
        ifstream ifs(filename, ios::binary);
        if (!ifs) {
            cerr << "Error: Failed to open file for reading: " << filename << endl;
            return;
        }
        recorder.deserializeDepth(ifs);
        ifs.close();
        cout << "Depth deserialized. Current depth: " << recorder.depth << endl;
    }


    void exp_RSMI(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points, vector<Point> insert_points, string model_path, string index_path)
    {

        RSMI *partition = new RSMI(0,  Constants::MAX_WIDTH);

        string key_path = Constants::KEY;
        file_utils::check_dir(key_path);
        string key_file = key_path + "paillier_" + to_string(exp_recorder.K) + ".key";
        
        // partition->save_key(key_file, paillier);

        
        partition->load_key(key_file, paillier);

        exp_recorder.clean();
        exp_recorder.structure_name = "RSMI";
        RSMI::model_path_root = model_path;
        RSMI::indices_path = index_path;

        cout<<"start build"<<endl;
        cout<<"point.size: "<<points.size()<<endl;

        //build index
        exp_recorder.depth = 0;
        auto start = chrono::high_resolution_clock::now();
        partition->model_path = model_path;
        partition->kmeans_build(exp_recorder, points, paillier);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        cout << "build time: " << exp_recorder.time << endl;
        exp_recorder.size = (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * exp_recorder.B + exp_recorder.B + Constants::DIM * Constants::DIM) * Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;
        file_writer.write_build(exp_recorder);
        exp_recorder.clean();
        
        cout<<"depth: "<<exp_recorder.depth<<endl;

        
        string serialize_path = index_path +"serialized_" + to_string(exp_recorder.N) +".rsmi";
        cout<<"path: "<<serialize_path<<endl;
        // partition->save_index(serialize_path);
       
        partition->load_index(serialize_path, paillier);
        

        
        string serialize_depth_path = index_path + "depth_"+ to_string(exp_recorder.N)  + ".bin";
       
        // serialize_depth(exp_recorder, serialize_depth_path);

        
        ExpRecorder load_recorder;
        deserialize_depth(exp_recorder, serialize_depth_path);

        // delete partition;

        cout<<"depth: "<<exp_recorder.depth<<endl;

        // cout<<"**********build index end**********"<<endl;



        exp_recorder.enc_depth = (exp_recorder.depth + 1) * float(0.2 * el);
        cout<<" enc_depth: "<<exp_recorder.enc_depth<<endl;
        //plaintext window query
        exp_recorder.time = 0;
        exp_recorder.window_size = areas[wl];
        exp_recorder.window_ratio = ratios[rl];
        exp_recorder.selectivity = selectivity[sl];

        // partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[wl]) + to_string(ratios[rl])]);
        partition->acc_window_query(exp_recorder, mbrs_map[to_string(selectivity[sl])]);
        cout << "RSMI::acc_window_query time: " << exp_recorder.time << endl;
        cout << "RSMI::acc_window_query page_access: " << exp_recorder.page_access << endl;
        exp_recorder.page_access = 0;
        exp_recorder.time = 0;

        cout<<"****************"<<endl;
        // partition->window_query(exp_recorder, mbrs_map[to_string(areas[wl]) + to_string(ratios[rl])]);
        partition->window_query(exp_recorder, mbrs_map[to_string(selectivity[sl])]);
        exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
        cout << "window_query time: " << exp_recorder.time << endl;
        cout << "window_query page_access: " << exp_recorder.page_access << endl;
        cout<< "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
        exp_recorder.page_access = 0;
        exp_recorder.time = 0;
        // exp_recorder.clean();
        cout<<"****************"<<endl;


        // baseline window query
        partition->Base_line_window_query(exp_recorder, mbrs_map[to_string(selectivity[sl])], paillier);
        exp_recorder.accuracy = ((double)exp_recorder.Base_line_window_query_result_size) / exp_recorder.acc_window_query_result_size;
        cout << "Base_line_window_query time: " << exp_recorder.time << endl;
        cout << "Base_line_window_query page_access: " << exp_recorder.page_access << endl;
        cout<< "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
        file_writer.write_baseline_window_query(exp_recorder);
        exp_recorder.page_access = 0;
        exp_recorder.time = 0;
        // exp_recorder.clean();
        cout<<"****************"<<endl;

        // secure window query
        partition->Secure_window_query(exp_recorder, mbrs_map[to_string(selectivity[sl])], paillier);
        exp_recorder.accuracy = ((double)exp_recorder.Secure_window_query_result_size) / exp_recorder.acc_window_query_result_size;
        cout << "Secure_window_query time: " << exp_recorder.time << endl;
        cout << "Secure_window_query page_access: " << exp_recorder.page_access << endl;
        cout<< "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
        file_writer.write_secure_window_query(exp_recorder);
        // exp_recorder.clean();
        exp_recorder.page_access = 0;
        exp_recorder.time = 0;
        cout<<"****************"<<endl<<endl<<endl;
        
        // //traver tree
        // partition->Order(exp_recorder);
        // cout<< "exp_recorder.net_class_num: " << exp_recorder.net_class_num << endl;
        
        
    }

    string RSMI::model_path_root = "";
    string RSMI::indices_path = "";

    int main(int argc, char **argv)
    {
        omp_set_num_threads(6);  // thread num

        int c;
        static struct option long_options[] =
        {
            {"cardinality", required_argument,NULL,'c'},
            {"distribution",required_argument,      NULL,'d'},
            {"skewness", required_argument,      NULL,'s'}
        };

        while(1)
        {
            int opt_index = 0;
            c = getopt_long(argc, argv,"c:d:s:", long_options,&opt_index);
            
            if(-1 == c)
            {
                break;
            }
            switch(c)
            {
                case 'c':
                    cardinality = atoll(optarg);
                    break;
                case 'd':
                    distribution = optarg;
                    break;
                case 's':
                    skewness = atoi(optarg);
                    break;
            }
        }

        ExpRecorder exp_recorder;
        exp_recorder.dataset_cardinality = cardinality;
        exp_recorder.distribution = distribution;
        exp_recorder.skewness = skewness;
        // inserted_num = cardinality / 2;

        inserted_num = cardinality * 0.1;

        // cout<<"inserted_num: "<<inserted_num<<endl;

        // TODO change filename
        string dataset_filename = Constants::DATASETS + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv";
        FileReader filereader(dataset_filename, ",");
       
        vector<Point> points = filereader.get_points();
        int dim = points[0].dim;
        cout<<"dim: "<<dim<<endl;
        exp_recorder.insert_num = inserted_num;
        exp_recorder.dim = dim;

        vector<Point> query_poitns;
        vector<Point> insert_points;
        //***********************write query data*********************
        FileWriter query_file_writer(Constants::QUERYPROFILES);
       
      
        query_poitns = Point::get_points(points, query_k_num);
       
        query_file_writer.write_points(query_poitns, exp_recorder);
        
        insert_points = Point::get_inserted_points(exp_recorder.insert_num, dim);
       
        query_file_writer.write_inserted_points(insert_points, exp_recorder);
        cout<<"widow_length: "<<window_length<<endl;
        cout<<"ratio_length: "<<ratio_length<<endl;
        //gennerate query windows
        for(int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = Mbr::get_mbrs(points, selectivity[i], query_window_num);
            cout << "mbrs.size(): " << mbrs.size() << endl;
            exp_recorder.selectivity = selectivity[i];
            query_file_writer.write_mbrs(mbrs, exp_recorder);
        }

       
        //**************************prepare  window query, and insertion data******************
        
        map<string, vector<Mbr>> mbrs_map;
        FileReader query_filereader;

        
        for (size_t i = 0; i < window_length; i++)
        {
            for (size_t j = 0; j < ratio_length; j++)
            {
                exp_recorder.window_size = areas[i];
                exp_recorder.window_ratio = ratios[j];
                vector<Mbr> mbrs = query_filereader.get_mbrs((Constants::QUERYPROFILES + Constants::WINDOW + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_" + to_string(exp_recorder.selectivity) +".csv"), ",");
                mbrs_map.insert(pair<string, vector<Mbr>>(to_string(areas[i]) + to_string(ratios[j]), mbrs));
            }
        }

        
        for (size_t i = 0; i < 5; i++)
        {
            exp_recorder.selectivity = selectivity[i];
            vector<Mbr> mbrs = query_filereader.get_mbrs((Constants::QUERYPROFILES + Constants::WINDOW + "D_" + to_string(exp_recorder.dim)+ "/" + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_" + to_string(exp_recorder.selectivity) + ".csv"), ",");
            mbrs_map.insert(pair<string, vector<Mbr>>(to_string(selectivity[i]), mbrs));
        }
        string model_root_path = Constants::TORCH_MODELS + distribution + "_" + to_string(cardinality);
        file_utils::check_dir(model_root_path);
        string model_path = model_root_path + "/";

        sl = 0;

        
        FileWriter file_writer(Constants::RECORDS);

        string indices_path = Constants::INDEX + distribution + "_" + to_string(cardinality) + "/" + "K_" + to_string(exp_recorder.K) + "_B_" + to_string(exp_recorder.B) + "_D_" + to_string(dim);
        file_utils::check_dir(indices_path);
        string index_path = indices_path + "/";
        cout << "N:" << exp_recorder.N << " K: " << exp_recorder.K << " B: " << exp_recorder.B << endl;
        cout << "leafnodenum:" << exp_recorder.leaf_node_num << endl;
        cout << "points.size(): " << points.size() << endl;
        auto points_1 = points;
        exp_RSMI(file_writer, exp_recorder, points_1, mbrs_map, query_poitns, insert_points, model_path, index_path);

        return 0;
    }
    #include <boost/serialization/export.hpp>
    
    // BOOST_CLASS_EXPORT(RSMI)
    BOOST_CLASS_EXPORT(NetClassifier)
    BOOST_CLASS_EXPORT(Net)
    BOOST_CLASS_EXPORT(Point)
    BOOST_CLASS_EXPORT(Mbr)
    // BOOST_CLASS_EXPORT(LeafNode)
    BOOST_CLASS_EXPORT(Node)
    // BOOST_CLASS_EXPORT(PaillierFast)
#endif  // use_gpu