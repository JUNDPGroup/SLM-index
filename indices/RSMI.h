#include <iostream>
#include <vector>
#include "../entities/Node.h"
#include "../entities/Point.h"
#include "../entities/Mbr.h"
#include "../entities/NonLeafNode.h"
#include "../entities/LeafNode.h"
#include <typeinfo>
#include "../utils/ExpRecorder.h"
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"
#include "../utils/Modelools_classifierNet.h"
#include "../curves/hilbert.H"
#include "../curves/hilbert4.H"
#include "../curves/z.H"
#include "../cluster_evaluator/clustering_evaluator.h"
#include "../cluster_evaluator/k-means_clustering.h"
// #include "../Paillier/paillier.h"
#include <ophelib/paillier_fast.h>
#include <ophelib/integer.h>
#include <map>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <unordered_set>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

#include "../agreements/SIC.h"
#include "../agreements/SM.h"
#include "../agreements/SSED.h"
#include "../agreements/SQ.h"
#include <gmp.h>
#include <gmpxx.h>
#include <random>
#include "../Serialize/serialization_helpers.hpp"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>  
#include <boost/serialization/unique_ptr.hpp>  
#include <boost/serialization/weak_ptr.hpp>    
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>

BOOST_SERIALIZATION_SHARED_PTR(Net) 
#include <boost/serialization/split_free.hpp>
BOOST_SERIALIZATION_SPLIT_FREE(mpz_class)

namespace boost {
namespace serialization {

template<class Archive>
void save(Archive& ar, const mpz_class& g, const unsigned int version) {
    std::string s = g.get_str();
    ar & s;
}

template<class Archive>
void load(Archive& ar, mpz_class& g, const unsigned int version) {
    std::string s;
    ar & s;
    g.set_str(s, 10);
}

} // namespace serialization
} // namespace boost

struct LeafNodeNoiseRecord {
    std::vector<Ciphertext> enc_mbr_low_noises;
    std::vector<Ciphertext> enc_mbr_high_noises;
    std::vector<std::vector<Ciphertext>> enc_point_coord_noises;
};

struct RSMINoiseRecord {
    Ciphertext enc_max_error_noise;
    Ciphertext enc_min_error_noise;
    Ciphertext enc_width_noise;
    Ciphertext enc_leaf_node_num_noise;
    Ciphertext enc_N_noise;

    std::vector<Ciphertext> enc_mbr_low_noises;
    std::vector<Ciphertext> enc_mbr_high_noises;

    std::map<int, std::shared_ptr<RSMINoiseRecord>> child_noises;
    std::vector<LeafNodeNoiseRecord> leafnode_noises;
};


using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace ophelib;
using namespace std;
Constants constants;
Integer SCALED(1000000);
enum NetType { REGRESSION_NET, CLASSIFICATION_NET };
class RSMI
{

private:
    int dim = 0;
    int level;             // no
    int index;             // no
    int max_partition_num; // wiat
    long long N = 0;       // wait
    int max_error = 0;     // yes
    int min_error = 0;     // yes
    int width = 0;         // yes
    int leaf_node_num = 0;     // yes

    bool is_last;

    int net_type;//0 represents regression, and 1 represents classification



    Ciphertext enc_max_error , enc_min_error;
    Ciphertext enc_width ;
    Ciphertext enc_leaf_node_num ;

    Ciphertext enc_is_last;
    Ciphertext enc_N;

    Mbr mbr;
    std::shared_ptr<Net> net;
    std::shared_ptr<NetClassifier> netclass;

    std::vector<__uint128_t> class_to_z; 
    std::vector<Ciphertext> enc_class_to_z; 





public:
    string model_path;
    string index_path;
    static string indices_path;
    static string model_path_root;
    map<int, RSMI> children;
    vector<LeafNode> leafnodes;
    vector<Ciphertext> enc_children_index;
    vector<Ciphertext> enc_leafnodes_index;
    
    virtual void printInfo() {
        std::cout << "This is a RSMI." << std::endl;
    }



    RSMI();

    RSMI(const RSMI &other)
    {
        
        dim = other.dim;
        level = other.level;
        index = other.index;
        max_partition_num = other.max_partition_num;
        N = other.N;
        max_error = other.max_error;
        min_error = other.min_error;
        width = other.width;
        leaf_node_num = other.leaf_node_num;
        is_last = other.is_last;
        net_type = other.net_type;
        enc_max_error = other.enc_max_error;
        enc_min_error = other.enc_min_error;
        enc_width = other.enc_width;
        enc_leaf_node_num = other.enc_leaf_node_num;
        enc_is_last = other.enc_is_last;
        enc_N = other.enc_N;
        mbr = other.mbr;

       
        if (other.net)
        {
            net = std::make_shared<Net>(*other.net);
        }

        
        if (other.netclass)
        {
            netclass = std::make_shared<NetClassifier>(*other.netclass);
        }

        class_to_z = other.class_to_z;
        enc_class_to_z = other.enc_class_to_z;

        model_path = other.model_path;
        index_path = other.index_path;


        for (const auto &kv : other.children)
        {
            children.emplace(kv.first, RSMI(kv.second));
        }

        
        leafnodes.clear();
        for (const auto &ln : other.leafnodes)
        {
            leafnodes.push_back(LeafNode(ln));
        }

        enc_children_index = other.enc_children_index;
        enc_leafnodes_index = other.enc_leafnodes_index;
    }

    RSMI(PaillierFast& paillier);
    RSMI(int index, int max_partition_num);
    RSMI(int index, int level, int max_partition_num);

    RSMI(int index, int level, int max_partition_num, PaillierFast& paillier);

    RSMI(int index, PaillierFast &paillier, int max_partition_num);

    RSMI(int index, PaillierFast &paillier, int max_partition_num, int &res);

    void init(PaillierFast& paillier);

    void encrypt_node(PaillierFast &paillier);

    void add_noise_to_leafnodes(vector<LeafNode>& leafnodes, PaillierFast& paillier,
                            vector<LeafNodeNoiseRecord>& noise_records);

    void add_noise_to_leafnode(LeafNode& leafnode, PaillierFast& paillier,
                            LeafNodeNoiseRecord& noise_records);

    void remove_noise_from_leafnode(LeafNode& leafnode, PaillierFast& paillier,
                                 const LeafNodeNoiseRecord& noise_records);
    void remove_noise_from_leafnodes(vector<LeafNode>& leafnodes, PaillierFast& paillier,
                                 const vector<LeafNodeNoiseRecord>& noise_records);

    void build(ExpRecorder &exp_recorder, vector<Point> points, PaillierFast& paillier);

    void Order(ExpRecorder &exp_recorder);

    void kmeans_build(ExpRecorder &exp_recorder, std::vector<Point> points, PaillierFast &paillier);

    void print_index_info(ExpRecorder &exp_recorder);

    bool Base_leafnodes_query(ExpRecorder &exp_recorder, Point query_point, LeafNode leafnode_1, PaillierFast& paillier);

    bool Secure_Base_leafnodes_query(ExpRecorder &exp_recorder, Point query_point, LeafNode leafnode_1, PaillierFast& paillier);

    shared_ptr<RSMINoiseRecord> add_noise_to_rsmi_recursive(RSMI& node, PaillierFast& paillier);

    void remove_noise_from_rsmi_recursive(RSMI& node, PaillierFast& paillier, const shared_ptr<RSMINoiseRecord>& record);

    shared_ptr<RSMINoiseRecord> add_noise_to_all_children(PaillierFast& paillier);

    void remove_noise_from_all_children(PaillierFast& paillier, const shared_ptr<RSMINoiseRecord>& record);

    void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    void Base_line_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows, PaillierFast& paillier);

    void Secure_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows, PaillierFast& paillier);
    
    void window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window);

    void Base_line_window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, PaillierFast& paillier);

    void Secure_window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, PaillierFast& paillier);

    void acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> acc_window_query(ExpRecorder &exp_recorder, Mbr query_windows);

    void insert(ExpRecorder &exp_recorder, Point, PaillierFast& paillier, string index_path);
    void insert(ExpRecorder &exp_recorder, vector<Point>, PaillierFast& paillier, string index_path);
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & dim;
        ar & level;
        ar & index;
        ar & max_partition_num;
        ar & N;
        ar & max_error;
        ar & min_error;
        ar & width;
        ar & leaf_node_num;
        ar & is_last;
        ar & net_type;
        ar & enc_max_error;
        ar & enc_min_error;
        ar & enc_width;
        ar & enc_leaf_node_num;
        ar & enc_is_last;
        ar & enc_N;
    
    
        ar & mbr;
        ar & net;
        ar & netclass;
        ar & class_to_z;
        ar & enc_class_to_z;
        ar & children;
        ar & leafnodes;
        ar & enc_children_index;
        ar & enc_leafnodes_index;
    }

    void save_index(const std::string &index_filename)
    {
        try
        {
            
            {
                std::ofstream index_ofs(index_filename, std::ios::binary);
                if (!index_ofs)
                {
                    throw std::runtime_error("cant write inedx to file: " + index_filename);
                }
                boost::archive::binary_oarchive index_oa(index_ofs);
                index_oa << *this;
                
            }

            std::cout << "success save index to : " << index_filename << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "save progress error: " << e.what() << std::endl;
           
            std::remove(index_filename.c_str());
        }
    }

    void save_key(const std::string &key_filename, PaillierFast &paillier)
    {
        try
        {
            
            {
                std::ofstream keypair_ofs(key_filename, std::ios::binary);
                if (!keypair_ofs)
                {
                    throw std::runtime_error("cant write key to file: " + key_filename);
                }
                boost::archive::binary_oarchive keypair_oa(keypair_ofs);
                const auto keypair = paillier.get_keypair();
                keypair_oa << keypair;
                
            }
            std::cout << "success save key to :" << key_filename << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "save progress error: " << e.what() << std::endl;
            
            std::remove(key_filename.c_str());
        }
    }

    
    void load_key(const std::string &key_filename, PaillierFast &paillier)
    {
        try
        {
            std::ifstream ifs(key_filename, std::ios::binary);
            if (!ifs)
            {
                throw std::runtime_error("cant read key from file " + key_filename);
            }

            boost::archive::binary_iarchive ia(ifs);

            
            ophelib::KeyPair keypair;
            ia >> keypair;

            
            paillier.~PaillierFast();
            if (keypair.priv.key_size_bits > 0)
            {
                new (&paillier) ophelib::PaillierFast(keypair.pub, keypair.priv);
            }
            else
            {
                new (&paillier) ophelib::PaillierFast(keypair.pub);
            }

            std::cout << "success from " << key_filename << " read key" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "read key has error: " << e.what() << std::endl;
            throw;
        }
    }
    void load_index(const std::string &index_filename, PaillierFast &paillier)
    {
        try
        {
            std::ifstream ifs(index_filename, std::ios::binary);
            if (!ifs)
            {
                throw std::runtime_error("cant read index from file : " + index_filename);
            }

            boost::archive::binary_iarchive ia(ifs);

           
            ia >> *this;

            
            set_paillier_context(paillier);

            std::cout << "success from  " << index_filename << " read index" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "read index has error: " << e.what() << std::endl;
            throw; 
        }
    }
    void set_paillier_context(PaillierFast& paillier) {
        auto n2_shared = paillier.get_n2();
        auto fast_mod = paillier.get_fast_mod();
        
       
        set_ciphertext_context(enc_max_error, n2_shared, fast_mod);
        set_ciphertext_context(enc_min_error, n2_shared, fast_mod);
        set_ciphertext_context(enc_width, n2_shared, fast_mod);
        set_ciphertext_context(enc_leaf_node_num, n2_shared, fast_mod);
        set_ciphertext_context(enc_is_last, n2_shared, fast_mod);
        set_ciphertext_context(enc_N, n2_shared, fast_mod);
        for(int i = 0; i < enc_class_to_z.size(); i++)
        {
            set_ciphertext_context(enc_class_to_z[i], n2_shared, fast_mod);
        }

        for(int i = 0; i < enc_children_index.size(); i++)
        {
            set_ciphertext_context(enc_children_index[i], n2_shared, fast_mod);
        }


        
        for (auto& child : children) {
            child.second.set_paillier_context(paillier);
        }
    
        for(int i = 0; i < enc_leafnodes_index.size(); i++)
        {
            set_ciphertext_context(enc_leafnodes_index[i], n2_shared, fast_mod);
        }

       
        for (auto& leaf : leafnodes) {
            for(int i = 0; i < leaf.mbr.dim; i++)
            {
                set_ciphertext_context(leaf.mbr.enc_lows[i], n2_shared, fast_mod);
                set_ciphertext_context(leaf.mbr.enc_highs[i], n2_shared, fast_mod);
            }
            for (auto &leaf : leafnodes)
            {
                for (auto &point : *(leaf.children))
                {
                    for (size_t j = 0; j < point.enc_coords.size(); ++j)
                    {
                        set_ciphertext_context(point.enc_coords[j], n2_shared, fast_mod);
                    }
                    set_ciphertext_context(point.enc_temp_dist, n2_shared, fast_mod);
                }
            }
        }
    }

    void set_ciphertext_context(ophelib::Ciphertext& cipher, 
                               std::shared_ptr<ophelib::Integer> n2_shared,
                               std::shared_ptr<ophelib::FastMod> fast_mod) {
        cipher.n2_shared = n2_shared;
        cipher.fast_mod = fast_mod;
    }
    std::string uint128_to_string(__uint128_t value)
    {
        if (value == 0)
            return "0";

        std::string result;
        while (value > 0)
        {
            result = std::to_string(static_cast<unsigned long long>(value % 1000000000)) + result;
            value /= 1000000000;
        }
        return result;
    }
};

RSMI::RSMI()
{
    // leafnodes = vector<LeafNode>(10);
}

RSMI::RSMI(PaillierFast& paillier)
{
    
    Integer zero(0);
    Ciphertext enc_0;
    enc_0 = paillier.encrypt(zero);
    max_error = 0, min_error = 0, width = 0, leaf_node_num = 0, N = 0;
    net = std::make_shared<Net>(dim);
    net->get_parameters();
    net->init_encrypted_params(paillier);
    encrypt_node(paillier);
    netclass = std::make_shared<NetClassifier>(dim, 256, 2);
    netclass->get_parameters();
    netclass->init_encrypted_params(paillier);
}

RSMI::RSMI(int index, int max_partition_num)
{
    this->index = index;
    this->max_partition_num = max_partition_num;
    this->level = 0;
}

RSMI::RSMI(int index, int level, int max_partition_num)
{
    this->index = index;
    this->level = level;
    this->max_partition_num = max_partition_num;
}




// void init_seed() {
//     srand(42);
//     std::mt19937 gen{42};
    
//     torch::manual_seed(42);              
//     torch::globalContext().setDeterministicCuDNN(true);
//     torch::globalContext().setBenchmarkCuDNN(false);
// #ifdef use_gpu
//     torch::cuda::manual_seed_all(42);    
// #endif
// }

void RSMI::init(PaillierFast& paillier)
{
    Integer zero(0);
    Ciphertext enc_0;
    enc_0 = paillier.encrypt(zero);
    enc_max_error = enc_0, enc_min_error = enc_0, enc_width = enc_0, enc_leaf_node_num = enc_0 ;
}

void RSMI::encrypt_node(PaillierFast& paillier)
{
    Integer one(1), zero(0);
    Ciphertext enc_1, enc_0;
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);
    Integer a(max_error), b(min_error), c(width), d(leaf_node_num), e(N);
    // if (b < 0)
    //     b = b + n;
    enc_max_error = paillier.encrypt(a);
    enc_min_error = paillier.encrypt(b);
    enc_width = paillier.encrypt(c);
    enc_leaf_node_num = paillier.encrypt(d);
    enc_N = paillier.encrypt(e);
    
    // enc_leafnodes_index.resize(leafnodes.size());
    // enc_children_index.resize(children.size());
    enc_leafnodes_index.clear();
    enc_children_index.clear();
    if (is_last)
    {
        for (int i = 0; i < leafnodes.size(); i++)
        {
            leafnodes[i].encrypt_leaf(paillier);
            enc_leafnodes_index.push_back(paillier.encrypt(i));
        }
    }
    mbr.encrypt_mbr(paillier);

    for(auto child: children)
    {
        enc_children_index.push_back(paillier.encrypt(child.first));
    }
}


int log4ceil(const int& N, const int  & B) {
    if (B <= 0 || N <= 0) return -1.0;
    return std::pow(2.0, std::floor(std::log(N / B) / std::log(4.0)));
}

void RSMI::Order(ExpRecorder &exp_recorder)
{
    // cout<<"acc_window_query:_level: "<<level<<endl;
    // mbr.print();
    vector<Point> window_query_results;
    if(!is_last)
    {
        if(net_type == 1) exp_recorder.net_class_num += netclass->num_classes;
        map<int, RSMI>::iterator iter = children.begin();
        while (iter != children.end())
        {
            iter->second.Order(exp_recorder);
            iter++;
        }
    }
    return;
}

void RSMI::kmeans_build(ExpRecorder &exp_recorder, std::vector<Point> points, PaillierFast &paillier) 
{
    
    dim = points[0].dim;
    // int page_size = constants.PAGESIZE;
    int page_size = exp_recorder.B;
    N = points.size();
    mbr = Mbr(dim);
    if (points.size() <= exp_recorder.N) {
        
        is_last = true;
        Integer is(1);
        enc_is_last = paillier.encrypt(is);
        this->model_path += "_" + std::to_string(level) + "_" + std::to_string(index);
        if (exp_recorder.depth < level) exp_recorder.depth = level;
        exp_recorder.last_level_model_num++;


        int D = dim;
        std::vector<std::vector<int>> ranks(D, std::vector<int>(points.size()));
        for (int d = 0; d < D; d++) {
            std::vector<std::pair<float, int>> tmp;
            for (int i = 0; i < points.size(); i++) tmp.emplace_back(points[i].coords[d], i);
            std::sort(tmp.begin(), tmp.end());
            for (int i = 0; i < tmp.size(); i++) ranks[d][tmp[i].second] = i;
        }
        for (int i = 0; i < points.size(); i++) {
            std::vector<bitmask_t> coord(D);
            for (int d = 0; d < D; d++) coord[d] = ranks[d][i];
            int bits = ceil(log2(points.size()));
            points[i].curve_val = hilbert_c2i(D, bits, coord.data());
        }
        std::sort(points.begin(), points.end(), sort_curve_val());

        width = N - 1;
        for (long i = 0; i < N; i++) points[i].index = (N == 1) ? 0 : i * 1.0 / (N - 1);

      
        leaf_node_num = points.size() / page_size;
        for (int i = 0; i < leaf_node_num; i++) {
            LeafNode leafNode(dim);
            auto bn = points.begin() + i * page_size;
            auto en = bn + page_size;
            leafNode.add_points(std::vector<Point>(bn, en));
            leafnodes.push_back(leafNode);
        }
        if (points.size() > page_size * leaf_node_num) {
            LeafNode leafNode(dim);
            auto bn = points.begin() + page_size * leaf_node_num;
            leafNode.add_points(std::vector<Point>(bn, points.end()));
            leafnodes.push_back(leafNode);
            leaf_node_num++;
        }
        exp_recorder.leaf_node_num += leaf_node_num;

        
        net_type = 0; 
        net = std::make_shared<Net>(dim, leaf_node_num / 2 + 2);
        std::vector<float> locations, labels;

        int max_cluster_size = 0;
        for (const auto& leaf : leafnodes) {
            max_cluster_size = std::max(max_cluster_size, (int)leaf.children->size());
        }

    
        for (const auto& leaf : leafnodes) {
            int cluster_size = leaf.children->size();
            int repeat_factor = std::ceil((float)max_cluster_size / cluster_size);
            repeat_factor = std::max(repeat_factor, 1); 

            // repeat_factor = 1;

            
            for (int i = 0; i < leaf.children->size(); i++) {
                
                Point p = *(leaf.children->begin() + i);
                for (int r = 0; r < repeat_factor; ++r) {
                    for (int d = 0; d < dim; ++d) {
                        locations.push_back(p.coords[d]);
                    }
                    labels.push_back(p.index); 
                }
                mbr.update(p);
                // mbr.print();
            }
        }

        auto start_train = chrono::high_resolution_clock::now();
        net->train_model(locations, labels);
        auto finish_train = chrono::high_resolution_clock::now();
        exp_recorder.train_time += chrono::duration_cast<chrono::nanoseconds>(finish_train - start_train).count();
        net->get_parameters();
        auto start = chrono::high_resolution_clock::now();
        net->init_encrypted_params(paillier);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.encrypt_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.res_net_num++;
        for (int i = 0; i < N; i++) {
            Point point = points[i];
            int predicted_index = (int)(net->predict(point) * leaf_node_num);
            predicted_index = predicted_index < 0 ? 0 : predicted_index;
            predicted_index = predicted_index >= leaf_node_num ? leaf_node_num - 1 : predicted_index;
            int error = i / page_size - predicted_index;
            if (error > 0) max_error = std::max(max_error, error);
            else min_error = std::min(min_error, error);
        }
        exp_recorder.average_max_error += max_error;
        exp_recorder.average_min_error += min_error;
        if (max_error - min_error > exp_recorder.max_error - exp_recorder.min_error) {
            exp_recorder.max_error = max_error;
            exp_recorder.min_error = min_error;
        }
        exp_recorder.RSMI_leaf_node_num++;
    } else {

        auto start_cluster = chrono::high_resolution_clock::now();
        is_last = false;
        Integer is(0);
        enc_is_last = paillier.encrypt(is);

        int D = dim;
        int bit_num = max_partition_num;
        // bit_num = log4ceil(exp_recorder.N, exp_recorder.B);
        cout<<"bitbum: "<<bit_num<<endl;
        long long total_cells = pow(bit_num, D);
        width = total_cells - 1;

        int clustering_score = evaluateClusteringDataset(points);
        bool use_kmeans = (clustering_score == 2);
        // bool use_kmeans = true;

        std::vector<float> locations;
        std::vector<float> labels;           
        std::vector<__uint128_t> class_labels; 
        std::map<int, std::vector<Point>> points_map;
        auto finish_cluster = chrono::high_resolution_clock::now();
        exp_recorder.cluster_time += chrono::duration_cast<chrono::nanoseconds>(finish_cluster - start_cluster).count();
        
        if (use_kmeans)
        {
            auto start_cluster = chrono::high_resolution_clock::now();
          
            net_type = 1; 

            KMeansResult result = perform_kmeans_clustering(points, 0, 10);

            std::vector<Point> centroids = result.centroids;
       
            bool fallback_to_classification = false;

            auto finish_cluster = chrono::high_resolution_clock::now();
            exp_recorder.cluster_time += chrono::duration_cast<chrono::nanoseconds>(finish_cluster - start_cluster).count();


            std::vector<std::vector<int>> ranks(D, std::vector<int>(centroids.size()));
            for (int d = 0; d < D; ++d)
            {
                std::vector<std::pair<float, int>> tmp;
                for (int i = 0; i < centroids.size(); ++i)
                    tmp.emplace_back(centroids[i].coords[d], i);
                std::sort(tmp.begin(), tmp.end());
                for (int i = 0; i < tmp.size(); ++i)
                    ranks[d][tmp[i].second] = i;
            }

            int bits = ceil(log2(max_partition_num));
            class_to_z.clear();
            enc_class_to_z.clear();
            for (int i = 0; i < centroids.size(); ++i)
            {
                std::vector<long long> coord(D);
                for (int d = 0; d < D; ++d)
                    coord[d] = ranks[d][i];
                __uint128_t z_val = compute_Z_value(coord.data(), D, bits);
                class_to_z.push_back(z_val);
                enc_class_to_z.push_back(paillier.encrypt((long long)z_val));
            }

            __uint128_t max_z = *std::max_element(class_to_z.begin(), class_to_z.end());
            width = static_cast<long long>(max_z); 
            std::vector<float> locations, labels;
            for (int i = 0; i < centroids.size(); ++i)
            {
                float norm_index = static_cast<float>(class_to_z[i]) / width;
                for (const Point &pt : result.clustered_points[i])
                {
                    for (int d = 0; d < D; ++d)
                        locations.push_back(pt.coords[d]);
                    labels.push_back(norm_index);
                    // mbr.update(points[i]);
                    // mbr.print();
                }
            }


            int max_cluster_size = 0;
            for (auto &kv : result.clustered_points)
                max_cluster_size = std::max(max_cluster_size, (int)kv.second.size());
            
            std::vector<float> oversampled_locations, oversampled_labels;
            for (int i = 0; i < centroids.size(); ++i) {
                const std::vector<Point>& cluster_pts = result.clustered_points[i];
                int cluster_size = cluster_pts.size();
                int repeat_factor = std::ceil((float)max_cluster_size / cluster_size);
                repeat_factor = std::max(repeat_factor, 1);

                // repeat_factor = 1;
                
                for (const Point& pt : cluster_pts) {
                    for (int r = 0; r < repeat_factor; ++r) {
                        for (int d = 0; d < D; ++d)
                            oversampled_locations.push_back(pt.coords[d]);
                        oversampled_labels.push_back(static_cast<float>(class_to_z[i]) / width);
                    }
                }
            }
            locations = oversampled_locations;
            labels = oversampled_labels;

            net_type = 0;
            net = std::make_shared<Net>(D);
            auto start_train = chrono::high_resolution_clock::now();
            net->train_model(locations, labels);
            auto finish_train = chrono::high_resolution_clock::now();
            exp_recorder.train_time += chrono::duration_cast<chrono::nanoseconds>(finish_train - start_train).count();
            net->get_parameters();
            // net->init_encrypted_params(paillier);

            std::map<int, std::vector<Point>> temp_map;
            for (const auto &pt : points)
            {
                int predicted_index = (int)(net->predict(pt) * width);
                predicted_index = std::max(0, std::min((int)width - 1, predicted_index));
                temp_map[predicted_index].push_back(pt);
            }

            int valid_clusters = 0;
            for (const auto &kv : temp_map)
                if (!kv.second.empty())
                    valid_clusters++;

            if (valid_clusters < 2)
            {
 
                fallback_to_classification = true;
            }
            else
            {
       
                points_map = temp_map;
                exp_recorder.res_net_num++;
                auto start = chrono::high_resolution_clock::now();
                net->init_encrypted_params(paillier);
                auto finish = chrono::high_resolution_clock::now();
                exp_recorder.encrypt_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
            }

            if (fallback_to_classification)
            {
                net_type = 1;
                width = centroids.size();


                int max_cluster_size = 0;
                for (auto &kv : result.clustered_points)
                    max_cluster_size = std::max(max_cluster_size, (int)kv.second.size());

                for (int i = 0; i < centroids.size(); ++i)
                {
                    const std::vector<Point> &cluster_pts = result.clustered_points[i];
                    cout << "z_values: " << (long long)class_to_z[i] << " cluster_pts.size(): " << cluster_pts.size() << endl;
                    for (const Point &pt : cluster_pts)
                    {
                        int repeat_factor = std::ceil((float)max_cluster_size / cluster_pts.size());
                        repeat_factor = std::max(repeat_factor, 1);

                        // repeat_factor = 1;
                        for (int r = 0; r < repeat_factor; ++r)
                        {
                            for (int d = 0; d < D; ++d)
                                locations.push_back(pt.coords[d]);
                            class_labels.push_back(i);
                        }

                        Point real_pt = pt;
                        points_map[static_cast<int>(class_to_z[i])].push_back(real_pt);
                        // mbr.update(real_pt);
                        // mbr.print();
                    }
                }

                cout << "class_labels.size()" << (long long)class_labels.size() << endl;
                for (int i = 0; i < class_to_z.size(); i++)
                    cout << "z_values: " << (long long)class_to_z[i] << endl;

    
                netclass = std::make_shared<NetClassifier>(dim, 256, centroids.size());
                bool retrain = true;
                int epoch = Constants::START_EPOCH;
               
                while (retrain)
                {
                    auto start_train = chrono::high_resolution_clock::now();
                    netclass->train_model_classification(locations, class_labels);
                    auto finish_train = chrono::high_resolution_clock::now();
                    exp_recorder.train_time += chrono::duration_cast<chrono::nanoseconds>(finish_train - start_train).count();

                    netclass->get_parameters();

                    auto start = chrono::high_resolution_clock::now();
                    netclass->init_encrypted_params(paillier);
                    auto finish = chrono::high_resolution_clock::now();
                    exp_recorder.encrypt_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();

                    std::map<int, std::vector<Point>> temp_map;
                    for (const auto &pt : points)
                    {
                        int predicted_class = static_cast<int>(netclass->predict(pt));
                        predicted_class = std::max(0, std::min((int)centroids.size() - 1, predicted_class));
                        int predicted_z = static_cast<int>(class_to_z[predicted_class]);
                        temp_map[predicted_z].push_back(pt);
                    }
                    int valid = 0;
                    for (const auto &kv : temp_map)
                    {
                        if (!kv.second.empty())
                        {
                            ++valid;
                            cout << "kv.second: " << kv.second.size() << endl;
                            cout << "kv.first: " << kv.first << endl;
                        }
                    }
                    retrain = (valid < 2);
                    if (!retrain)
                    {
                        points_map = temp_map;
                        exp_recorder.class_net_num++;
                    }
                    else
                        epoch += Constants::EPOCH_ADDED;
                }
      
            }
            for(int i = 0; i < points.size(); ++i)
            {
                mbr.update(points[i]);
                // mbr.print();
            }
        }

        else {
            
          
            net_type = 0;

           
            std::vector<std::vector<int>> ranks(D, std::vector<int>(points.size()));
            for (int d = 0; d < D; d++) {
                std::vector<std::pair<float, int>> tmp;
                for (int i = 0; i < points.size(); i++) tmp.emplace_back(points[i].coords[d], i);
                std::sort(tmp.begin(), tmp.end());
                for (int i = 0; i < tmp.size(); i++) ranks[d][tmp[i].second] = i;
            }

            int bits = ceil(log2(bit_num));
            for (int i = 0; i < points.size(); ++i) {
                long long *coords = new long long[D];
                for (int d = 0; d < D; d++) coords[d] = ranks[d][i];
                __uint128_t z_val = compute_Z_value(coords, D, bits);
                delete[] coords;

                points[i].index = static_cast<float>(z_val) / width; // 归一化 label
                points_map[static_cast<int>(z_val)].push_back(points[i]);
                mbr.update(points[i]);
                // mbr.print();
            }

           
            bool retrain = true;
            int epoch = Constants::START_EPOCH;
            
            while (retrain) {
                net = std::make_shared<Net>(D);

                
                int max_cluster_size = 0;
                for (const auto& kv : points_map) {
                    max_cluster_size = std::max(max_cluster_size, (int)kv.second.size());
                }

                
                std::vector<float> locations, labels;
                for (const auto& kv : points_map) {
                    int cluster_size = kv.second.size();
                    int repeat_factor = std::ceil((float)max_cluster_size / cluster_size);
                    repeat_factor = std::max(repeat_factor, 1);
                    // repeat_factor = 1;

                    for (const Point& pt : kv.second) {
                       
                        for (int r = 0; r < repeat_factor; ++r) {
                            for (int d = 0; d < D; ++d) {
                                locations.push_back(pt.coords[d]);
                            }
                            labels.push_back(pt.index);
                        }
                    }
                }

                auto start_train = chrono::high_resolution_clock::now();
                net->train_model(locations, labels);
                auto finish_train = chrono::high_resolution_clock::now();
                exp_recorder.train_time += chrono::duration_cast<chrono::nanoseconds>(finish_train - start_train).count();
                net->get_parameters();

                auto start = chrono::high_resolution_clock::now();
                net->init_encrypted_params(paillier);
                auto finish = chrono::high_resolution_clock::now();
                exp_recorder.encrypt_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();

                std::map<int, std::vector<Point>> temp_map;
                for (const auto& pt : points) {
                    int predicted_index = static_cast<int>(net->predict(pt) * width);
                    predicted_index = std::max(0, std::min(width - 1, predicted_index));
                    temp_map[predicted_index].push_back(pt);
                }

                int valid = 0;
                for (const auto &kv : temp_map)
                {
                    if (!kv.second.empty()) 
                    {
                        ++valid;
                        cout<<"kv.second: "<<kv.second.size()<<endl;
                        cout<<"kv.first: "<<kv.first<<endl;
                    }
                }
                retrain = (valid < 2);
                if (!retrain) 
                {
                    points_map = temp_map;
                    exp_recorder.res_net_num++;
                }
                else epoch += Constants::EPOCH_ADDED;
            }
          
        }

        // === 递归构建子节点 ===
        for (auto &kv : points_map) {
            if (!kv.second.empty()) {
                RSMI partition(kv.first, level + 1, max_partition_num);
                partition.model_path = model_path;
                partition.kmeans_build(exp_recorder, kv.second, paillier);
                children.emplace(kv.first, std::move(partition));
            }
        }

        exp_recorder.non_leaf_node_num++;
    }


    auto start = chrono::high_resolution_clock::now();
    encrypt_node(paillier);
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.encrypt_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
}









void RSMI::print_index_info(ExpRecorder &exp_recorder)
{
    cout << "finish point_query max_error: " << exp_recorder.max_error << endl;
    cout << "finish point_query min_error: " << exp_recorder.min_error << endl;
    cout << "finish point_query average_max_error: " << exp_recorder.average_max_error << endl;
    cout << "finish point_query average_min_error: " << exp_recorder.average_min_error << endl;
    cout << "last_level_model_num: " << exp_recorder.last_level_model_num << endl;
    cout << "leaf_node_num: " << exp_recorder.leaf_node_num << endl;
    cout << "non_leaf_node_num: " << exp_recorder.non_leaf_node_num << endl;
    cout << "depth: " << exp_recorder.depth << endl;
}


bool RSMI::Base_leafnodes_query(ExpRecorder &exp_recorder, Point query_point, LeafNode leafnode, PaillierFast& paillier)
{
    Integer one(1), zero(0);
    Ciphertext enc_1, enc_0;
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);
    // if (leafnode_0.mbr.enc_contains(query_point, paillier) || leafnode_1.mbr.enc_contains(query_point, paillier))
    if (leafnode.mbr.enc_contains(query_point, paillier))
    {
        cout<<"start"<<endl;
        exp_recorder.page_access += 1;

        auto it = leafnode.children->begin();

        int is_true = 0;
        // DAP
        Integer a, b;
        cout<<"0"<<endl;
        for (Point point : (*leafnode.children))
        {
            is_true = 0;
            
            for(int j = 0; j < dim; j++)
            {
                a = paillier.decrypt(point.enc_coords[j]);
                b = paillier.decrypt(query_point.enc_coords[j]);
                cout<<"a: "<<a<<' '<<"b: "<<b<<endl; 
                if(a != b) is_true++; 
            }
            if(!is_true)
            {
                cout << "true" << endl;
                exp_recorder.result++;
                return true;
            }
        }
        // // DSP
        // if (is_true != dim)
        // {
        //     cout << "true" << endl;
        //     exp_recorder.result++;
        //     return true;
        // }
    }
    return false;
}

bool RSMI::Secure_Base_leafnodes_query(ExpRecorder &exp_recorder, Point query_point, LeafNode leafnode, PaillierFast& paillier)
{
    Integer one(1), zero(0);
    Ciphertext enc_1, enc_0;
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);
    // if (leafnode_0.mbr.enc_contains(query_point, paillier) || leafnode_1.mbr.enc_contains(query_point, paillier))
    if (leafnode.mbr.enc_contains(query_point, paillier))
    {
        cout<<"start"<<endl;
        exp_recorder.page_access += 1;

        auto it = leafnode.children->begin();

        int is_true = 0;
        // DAP
        Integer a, b;
        cout<<"0"<<endl;
        for (Point point : (*leafnode.children))
        {
            is_true = 0;
            
            for(int j = 0; j < dim; j++)
            {
                a = paillier.decrypt(point.enc_coords[j]);
                b = paillier.decrypt(query_point.enc_coords[j]);
                cout<<"a: "<<a<<' '<<"b: "<<b<<endl; 
                if(a != b) is_true++; 
            }
            if(!is_true)
            {
                cout << "true" << endl;
                exp_recorder.result++;
                return true;
            }
        }
        // // DSP
        // if (is_true != dim)
        // {
        //     cout << "true" << endl;
        //     exp_recorder.result++;
        //     return true;
        // }
    }
    return false;
}



int generate_noise_int(int low = 0, int high = 1000) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(low, high);
    return dis(gen);
}


void RSMI::add_noise_to_leafnodes(vector<LeafNode>& leafnodes, PaillierFast& paillier,
                            vector<LeafNodeNoiseRecord>& noise_records) {
    noise_records.clear();
    noise_records.resize(leafnodes.size());
    // cout<<"leafnodes.size(): "<<leafnodes.size()<<endl;
    for (int i = 0; i < leafnodes.size(); ++i) {
        LeafNode& node = leafnodes[i];
        LeafNodeNoiseRecord& record = noise_records[i];
        // int dim = node.dim;
        // cout<<"dim: "<<dim<<endl;
       
        // cout<<"jiazaombr"<<endl;
        for (int d = 0; d < dim; ++d) {
            int r_low = generate_noise_int();
            int r_high = generate_noise_int();

            // cout<<"r_low: "<<r_low<<endl;
            // cout<<"r_high: "<<r_high<<endl;

            Ciphertext enc_r_low = paillier.encrypt(r_low);
            Ciphertext enc_r_high = paillier.encrypt(r_high);

            node.mbr.enc_lows[d] += enc_r_low;
            node.mbr.enc_highs[d] += enc_r_high;

            record.enc_mbr_low_noises.push_back(enc_r_low);
            record.enc_mbr_high_noises.push_back(enc_r_high);
        }

    
        // cout<<"jiazaopoint"<<endl;
        for (auto& point : *(node.children)) {
            std::vector<Ciphertext> enc_coord_noise;
            for (int d = 0; d < dim; ++d) {
                int r = generate_noise_int();
                Ciphertext enc_r = paillier.encrypt(r);
                point.enc_coords[d] += enc_r;
                enc_coord_noise.push_back(enc_r);
            }
            record.enc_point_coord_noises.push_back(enc_coord_noise);
        }
    }
}

void RSMI::add_noise_to_leafnode(LeafNode& leafnode, PaillierFast& paillier,
                            LeafNodeNoiseRecord& noise_records) {
    // noise_records.clear();
    // noise_records.resize(leafnodes.size());

    // for (int i = 0; i < leafnodes.size(); ++i) {
    LeafNode& node = leafnode;
    LeafNodeNoiseRecord &record = noise_records;
    // if(node.children->size() == 0) return ;
    // int dim = node.dim;


    for (int d = 0; d < dim; ++d)
    {
        int r_low = generate_noise_int();
        int r_high = generate_noise_int();

        Ciphertext enc_r_low = paillier.encrypt(r_low);
        Ciphertext enc_r_high = paillier.encrypt(r_high);

        node.mbr.enc_lows[d] += enc_r_low;
        node.mbr.enc_highs[d] += enc_r_high;

        record.enc_mbr_low_noises.push_back(enc_r_low);
        record.enc_mbr_high_noises.push_back(enc_r_high);
    }


    for (auto &point : *(node.children))
    {
        std::vector<Ciphertext> enc_coord_noise;
        for (int d = 0; d < dim; ++d)
        {
            int r = generate_noise_int();
            Ciphertext enc_r = paillier.encrypt(r);
            point.enc_coords[d] += enc_r;
            enc_coord_noise.push_back(enc_r);
        }
        record.enc_point_coord_noises.push_back(enc_coord_noise);
    }
    // }
}

void RSMI::remove_noise_from_leafnode(LeafNode& leafnode, PaillierFast& paillier,
                                 const LeafNodeNoiseRecord& noise_records) {
    // for (int i = 0; i < leafnodes.size(); ++i) {
    LeafNode &node = leafnode;
    const LeafNodeNoiseRecord &record = noise_records;
    // if(node.children->size() == 0) return ;
    // int dim = node.dim;

    // cout<<"quzaombr"<<endl;
    // node.mbr.print();
    cout<<node.children->size()<<endl;
    for (int d = 0; d < dim; ++d)
    {
        node.mbr.enc_lows[d] -= record.enc_mbr_low_noises[d];
        node.mbr.enc_highs[d] -= record.enc_mbr_high_noises[d];
    }

    // cout<<"quzaopoint"<<endl;
    for (int j = 0; j < node.children->size(); ++j)
    {
        Point &point = (*(node.children))[j];
        for (int d = 0; d < dim; ++d)
        {
            point.enc_coords[d] -= record.enc_point_coord_noises[j][d];
        }
    }
    // }
}

void RSMI::remove_noise_from_leafnodes(vector<LeafNode>& leafnodes, PaillierFast& paillier,
                                 const vector<LeafNodeNoiseRecord>& noise_records) {
    for (int i = 0; i < leafnodes.size(); ++i) {
        LeafNode& node = leafnodes[i];
        const LeafNodeNoiseRecord& record = noise_records[i];
        // if(node.children->size() == 0) return ;
        // int dim = node.dim;

       
        // cout<<"quzaombr"<<endl;
        for (int d = 0; d < dim; ++d) {
            node.mbr.enc_lows[d] -= record.enc_mbr_low_noises[d];
            node.mbr.enc_highs[d] -= record.enc_mbr_high_noises[d];
        }

       
        // cout<<"quzaopoint"<<endl;
        for (int j = 0; j < node.children->size(); ++j) {
            Point& point = (*(node.children))[j];
            for (int d = 0; d < dim; ++d) {
                point.enc_coords[d] -= record.enc_point_coord_noises[j][d];
            }
        }
    }
}



std::shared_ptr<RSMINoiseRecord> RSMI::add_noise_to_rsmi_recursive(RSMI& node, PaillierFast& paillier)
{
    auto record = std::make_shared<RSMINoiseRecord>();
    // int dim = node.mbr.dim;

    
    int r1 = generate_noise_int(), r2 = generate_noise_int(), r3 = generate_noise_int(), r4 = generate_noise_int();
    long long r5 = generate_noise_int();

    record->enc_max_error_noise = paillier.encrypt(r1);
    record->enc_min_error_noise = paillier.encrypt(r2);
    record->enc_width_noise = paillier.encrypt(r3);
    record->enc_leaf_node_num_noise = paillier.encrypt(r4);
    record->enc_N_noise = paillier.encrypt(r5);

    node.enc_max_error += record->enc_max_error_noise;
    node.enc_min_error += record->enc_min_error_noise;
    node.enc_width += record->enc_width_noise;
    node.enc_leaf_node_num += record->enc_leaf_node_num_noise;
    node.enc_N += record->enc_N_noise;

    cout<<"jiazaombr"<<endl;
    
    cout<<"dim: "<<dim<<endl;
    cout<<"enc_lows.size(): "<<node.mbr.enc_lows.size()<<" enc_highs.size(): "<<node.mbr.enc_highs.size()<<endl;
    for (int d = 0; d < dim; ++d) {
        int r_low = generate_noise_int(), r_high = generate_noise_int();
        Ciphertext enc_r_low = paillier.encrypt(r_low);
        Ciphertext enc_r_high = paillier.encrypt(r_high);

        node.mbr.enc_lows[d] += enc_r_low;
        node.mbr.enc_highs[d] += enc_r_high;

        record->enc_mbr_low_noises.push_back(enc_r_low);
        record->enc_mbr_high_noises.push_back(enc_r_high);
    }

    if(net_type == 0)
    {
       
        cout<<"net"<<endl;
        for(int d=0; d < dim; d++) {
            for(int i=0; i<net->width; ++i) {
                Ciphertext enc_1 = paillier.encrypt(0);
                net->enc_w1_d[d][i] = net->enc_w1_d[d][i] + enc_1;
            }
        }
    

        for(int i=0; i<net->width; ++i) {
            Ciphertext enc_1 = paillier.encrypt(0);
            net->enc_b1[i] = net->enc_b1[i] + enc_1;
            net->enc_w2[i] = net->enc_w2[i] + enc_1;
        }
    
     
        Ciphertext enc_1 = paillier.encrypt(0);
        net->enc_b2 = net->enc_b2 + enc_1;
    }
    else if(net_type == 1)
    {
        cout<<"netclass"<<endl;
        for (int d = 0; d < dim; ++d)
        {
            for (int i = 0; i < netclass->width; ++i)
            {
                Ciphertext enc_1 = paillier.encrypt(0);    
                netclass->enc_w1_d[d][i] = netclass->enc_w1_d[d][i] + enc_1;
            }
        }
        // Ciphertext enc_1 = paillier.encrypt(1);
        for (int i = 0; i < netclass->width; ++i)
        {
            Ciphertext enc_1 = paillier.encrypt(0);
            netclass->enc_b1[i] = netclass->enc_b1[i] + enc_1;
        }


        for (int c = 0; c < netclass->num_classes; ++c) {
            for (int i = 0; i < netclass->width; ++i)
            {
                Ciphertext enc_1 = paillier.encrypt(0);
                netclass->enc_w2_by_class[c][i] = netclass->enc_w2_by_class[c][i] + enc_1;
            }
            Ciphertext enc_1 = paillier.encrypt(0);
            netclass->enc_b2_by_class[c] = netclass->enc_b2_by_class[c] + enc_1;
        }
    }


    if (!node.is_last) {
        for (std::map<int, RSMI>::iterator it = node.children.begin(); it != node.children.end(); ++it) {
            int id = it->first;
            RSMI& child_node = it->second;
            auto child_record = add_noise_to_rsmi_recursive(child_node, paillier);
            record->child_noises[id] = child_record;
        }
    }

    if (node.is_last) {
        record->leafnode_noises.resize(node.leafnodes.size());
        for (int i = 0; i < node.leafnodes.size(); ++i) {
            LeafNode& leaf = node.leafnodes[i];
            if(leaf.children->size() == 0) continue;
            LeafNodeNoiseRecord& leaf_rec = record->leafnode_noises[i];

            for (int d = 0; d < dim; ++d) {
                int r_low = generate_noise_int(), r_high = generate_noise_int();
                Ciphertext enc_r_low = paillier.encrypt(r_low);
                Ciphertext enc_r_high = paillier.encrypt(r_high);
                leaf.mbr.enc_lows[d] += enc_r_low;
                leaf.mbr.enc_highs[d] += enc_r_high;
                leaf_rec.enc_mbr_low_noises.push_back(enc_r_low);
                leaf_rec.enc_mbr_high_noises.push_back(enc_r_high);
            }

            for (std::vector<Point>::iterator pt = leaf.children->begin(); pt != leaf.children->end(); ++pt) {
                std::vector<Ciphertext> coord_noise;
                for (int d = 0; d < dim; ++d) {
                    int r = generate_noise_int();
                    Ciphertext enc_r = paillier.encrypt(r);
                    pt->enc_coords[d] += enc_r;
                    coord_noise.push_back(enc_r);
                }
                leaf_rec.enc_point_coord_noises.push_back(coord_noise);
            }
        }
    }

    return record;
}
void RSMI::remove_noise_from_rsmi_recursive(RSMI &node, PaillierFast &paillier, const std::shared_ptr<RSMINoiseRecord> &record)
{
    // int dim = node.mbr.dim;
    node.enc_max_error -= record->enc_max_error_noise;
    node.enc_min_error -= record->enc_min_error_noise;
    node.enc_width -= record->enc_width_noise;
    node.enc_leaf_node_num -= record->enc_leaf_node_num_noise;
    node.enc_N -= record->enc_N_noise;

    for (int d = 0; d < dim; ++d)
    {
        node.mbr.enc_lows[d] -= record->enc_mbr_low_noises[d];
        node.mbr.enc_highs[d] -= record->enc_mbr_high_noises[d];
    }

    if(net_type == 0)
    {
        Ciphertext enc_1 = paillier.encrypt(0);
      
        for(int d=0; d < dim; d++) {
            for(int i=0; i<net->width; ++i) {
                net->enc_w1_d[d][i] = net->enc_w1_d[d][i] - enc_1;
            }
        }
    
        
        for(int i=0; i<net->width; ++i) {
            net->enc_b1[i] = net->enc_b1[i] - enc_1;
            net->enc_w2[i] = net->enc_w2[i] - enc_1;
        }
    
        net->enc_b2 = net->enc_b2 - enc_1;
    }
    else if(net_type == 1)
    {
        Ciphertext enc_1 = paillier.encrypt(0);
        for (int d = 0; d < dim; ++d)
        {
            for (int i = 0; i < netclass->width; ++i)
            {    
                netclass->enc_w1_d[d][i] = netclass->enc_w1_d[d][i] - enc_1;
            }
        }
        for (int i = 0; i < netclass->width; ++i)
            netclass->enc_b1[i] = netclass->enc_b1[i] - enc_1;

 

        for (int c = 0; c < netclass->num_classes; ++c) {
            for (int i = 0; i < netclass->width; ++i)
                netclass->enc_w2_by_class[c][i] = netclass->enc_w2_by_class[c][i] - enc_1;
            netclass->enc_b2_by_class[c] = netclass->enc_b2_by_class[c] - enc_1;
        }
    }
    if (!node.is_last)
    {
        if(node.children.size() == 0) return;
        for (std::map<int, RSMI>::iterator it = node.children.begin(); it != node.children.end(); ++it)
        {
            int id = it->first;
            RSMI &child_node = it->second;
            std::map<int, std::shared_ptr<RSMINoiseRecord>>::const_iterator rit = record->child_noises.find(id);
            if (rit != record->child_noises.end())
            {
                remove_noise_from_rsmi_recursive(child_node, paillier, rit->second);
            }
        }
    }

    if (node.is_last)
    {
        for (int i = 0; i < node.leafnodes.size(); ++i)
        {
            LeafNode &leaf = node.leafnodes[i];
            const LeafNodeNoiseRecord &leaf_rec = record->leafnode_noises[i];
            if(leaf.children->size() == 0) continue;
            for (int d = 0; d < dim; ++d)
            {
                leaf.mbr.enc_lows[d] -= leaf_rec.enc_mbr_low_noises[d];
                leaf.mbr.enc_highs[d] -= leaf_rec.enc_mbr_high_noises[d];
            }

            for (int j = 0; j < leaf.children->size(); ++j)
            {
                Point &pt = (*leaf.children)[j];
                for (int d = 0; d < dim; ++d)
                {
                    pt.enc_coords[d] -= leaf_rec.enc_point_coord_noises[j][d];
                }
            }
        }
    }
}
shared_ptr<RSMINoiseRecord> RSMI::add_noise_to_all_children(PaillierFast& paillier){
    auto record = std::make_shared<RSMINoiseRecord>();

    for (auto it = this->children.begin(); it != this->children.end(); ++it) {
        int id = it->first;
        RSMI& child_node = it->second;
        cout<<"jiazaorsmi"<<endl;
        auto child_record = add_noise_to_rsmi_recursive(child_node, paillier);
        record->child_noises[id] = child_record;
    }

    return record;
}

void RSMI::remove_noise_from_all_children(PaillierFast& paillier, const std::shared_ptr<RSMINoiseRecord>& record) {
    for (auto it = this->children.begin(); it != this->children.end(); ++it) {
        int id = it->first;
        RSMI& child_node = it->second;

        auto rec_it = record->child_noises.find(id);
        if (rec_it != record->child_noises.end()) {
            remove_noise_from_rsmi_recursive(child_node, paillier, rec_it->second);
        }
    }
}



void RSMI::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    long long time_cost = 0;
    int length = query_windows.size();

    for (int i = 0; i < length; i++)
    {

        vector<Point> vertexes = query_windows[i].get_corner_points();
        query_windows[i].print();
        auto start = chrono::high_resolution_clock::now();
        // cout<<"start"<<endl;
        window_query(exp_recorder, vertexes, query_windows[i]);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.window_query_result_size += exp_recorder.window_query_results.size();
        exp_recorder.window_query_results.clear();
        exp_recorder.window_query_results.shrink_to_fit();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    exp_recorder.time /= length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
    cout<<"exp_recorder.window_query_result_size: "<<exp_recorder.window_query_result_size<<endl;
}

void RSMI::Base_line_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows, PaillierFast& paillier)
{
    long long time_cost = 0;
    int length = query_windows.size();

    for (int i = 0; i < length; i++)
    {
        cout<<"Base_line_i: "<<i<<endl;
        query_windows[i].encrypt_mbr(paillier);
        vector<Point> vertexes = query_windows[i].init_get_corner_points();
        query_windows[i].print();
        auto start = chrono::high_resolution_clock::now();
        Base_line_window_query(exp_recorder, vertexes, query_windows[i], paillier);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.Base_line_window_query_result_size += exp_recorder.Base_line_window_query_results.size();
        exp_recorder.Base_line_window_query_results.clear();
        exp_recorder.Base_line_window_query_results.shrink_to_fit();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        if(i == 0) exp_recorder.min_time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        else exp_recorder.min_time = min(exp_recorder.min_time, chrono::duration_cast<chrono::nanoseconds>(finish - start).count());
    }
    exp_recorder.time /= length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
    cout<<"exp_recorder.Base_line_window_query_result_size: "<<exp_recorder.Base_line_window_query_result_size<<endl;
}


void RSMI::Secure_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows, PaillierFast& paillier)
{
    long long time_cost = 0;
    int length = query_windows.size();
    // length = min(3, length);
    for (int i = 0; i < length; i++)
    {
        cout<<"Secure_i: "<<i<<endl;
        query_windows[i].encrypt_mbr(paillier);
        vector<Point> vertexes = query_windows[i].init_get_corner_points();
        auto start = chrono::high_resolution_clock::now();
        Secure_window_query(exp_recorder, vertexes, query_windows[i], paillier);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.Secure_window_query_result_size += exp_recorder.Secure_window_query_results.size();

        exp_recorder.Secure_window_query_results.clear();
        exp_recorder.Secure_window_query_results.shrink_to_fit();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        if(i == 0) exp_recorder.min_time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        else exp_recorder.min_time = min(exp_recorder.min_time, chrono::duration_cast<chrono::nanoseconds>(finish - start).count());
    }
    exp_recorder.time /= length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
    cout<<"exp_recorder.Secure_window_query_result_size: "<<exp_recorder.Secure_window_query_result_size<<endl;
}


void RSMI::window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window)
{
    // mbr.print();
    if (is_last)
    {
        int leafnodes_size = leafnodes.size();
        int front = leafnodes_size - 1;
        int back = 0;
        if (leaf_node_num == 0)
        {
            // cout<<"0"<<endl;
            return;
        }
        else if (leaf_node_num < 2)
        {
            // cout<<"1"<<endl;
            front = 0;
            back = 0;
        }
        else
        {
            // cout<<"2"<<endl;
            int max = 0;
            int min = width;
            for (size_t i = 0; i < vertexes.size(); i++)
            {
                int predicted_index = 0;
                if(net_type == 0)
                {
                    predicted_index = net->predict(vertexes[i]) * leaf_node_num;

                    predicted_index = predicted_index < 0 ? 0 : predicted_index;
                    predicted_index = predicted_index > width ? width : predicted_index;
                }
                else if(net_type == 1)
                {
                    predicted_index = netclass->predict(vertexes[i]);
                    predicted_index = predicted_index < 0 ? 0 : predicted_index;
                    predicted_index = predicted_index >= width ? width - 1 : predicted_index;
                    // cout<<"predicted_index: "<<predicted_index<<endl;
                    predicted_index = static_cast<int>(class_to_z[predicted_index]);
                    // cout<<"predicted_index: "<<predicted_index<<endl;
                }
                int predicted_index_max = predicted_index + max_error;
                int predicted_index_min = predicted_index + min_error;
                if (predicted_index_min < min)
                {
                    min = predicted_index_min;
                }
                if (predicted_index_max > max)
                {
                    max = predicted_index_max;
                }
            }

            front = min < 0 ? 0 : min;
            back = max >= leafnodes_size ? leafnodes_size - 1 : max;
        }

        for (size_t i = front; i <= back; i++)
        {
            LeafNode leafnode = leafnodes[i];

            if (leafnode.mbr.interact(query_window))
            {
                exp_recorder.page_access += 1;
                // cout<<"page_access: "<<exp_recorder.page_access<<endl;
                for (Point point : (*leafnode.children))
                {
                    if (query_window.contains(point))
                    {
                        exp_recorder.window_query_results.push_back(point);
                        // exp_recorder.window_query_result_size++;
                    }
                }
            }
        }
        return;
    }
    else
    {
        int children_size = width;
        int front = children_size - 1;
        int back = 0;
        for (size_t i = 0; i < vertexes.size(); i++)
        {
            int predicted_index = 0;
            if(net_type == 0)
            {
                predicted_index = net->predict(vertexes[i]) * children_size;
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
            }
            else if(net_type == 1)
            {
                predicted_index = netclass->predict(vertexes[i]);
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
                predicted_index = static_cast<int>(class_to_z[predicted_index]);
            }
            if (predicted_index < front)
            {
                front = predicted_index;
            }
            if (predicted_index > back)
            {
                back = predicted_index;
            }
        }


        for (size_t i = front; i <= back; i++)
        {
            if (children.count(i) == 0)
            {
                continue;
            }
            // mbr.print();
            // query_window.print();
            if (children[i].mbr.interact(query_window))
            {
                // children[i].mbr.print();
                children[i].window_query(exp_recorder, vertexes, query_window);
            }
        }
    }
}

void RSMI::Base_line_window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, PaillierFast& paillier)
{
    // mbr.print();
    Ciphertext enc_1, enc_0;
    Integer one(1), zero(0);
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);
    // query_window.encrypt_mbr(paillier);
    if (SICrun(enc_1, enc_is_last, paillier) == 1)
    {
        Integer denc_leafnodes_size(leafnodes.size());
        Ciphertext leafnodes_size;
        leafnodes_size = paillier.encrypt(denc_leafnodes_size);
        Ciphertext front;
        front = leafnodes_size - enc_1;
        Ciphertext back = enc_0;
        Ciphertext enc_2;
        Integer two(2);
        enc_2 = paillier.encrypt(two);
        if(SICrun(enc_leaf_node_num, enc_0, paillier) == 1) return;
        else if(SICrun(enc_2, enc_leaf_node_num, paillier) == 0)
        {
            // cout<<2<<endl;
            front = enc_0;
            back = enc_0;
        }
        else 
        {
            // cout<<3<<endl;
            Ciphertext max = enc_0, min = enc_width;
            // #pragma omp parallel for
            for(size_t i = 0; i < vertexes.size(); i++)
            {
                Ciphertext predicted_index;
                if(net_type == 0)
                {
                    predicted_index = net->enc_predict(vertexes[i], paillier);

                    Integer flag;
                    flag = SICrun(enc_0, predicted_index, paillier);
                    if (flag == 0)
                        predicted_index = enc_0;

                    PublicKey pub = paillier.get_pub();
                    Integer Random, denc_R(generateRandomNumber()), n(pub.n);
                    Ciphertext enc_R;
                    Random = denc_R * (SCALED * SCALED * SCALED);
                    enc_R = paillier.encrypt(Random);
                    predicted_index = predicted_index + enc_R;

                    // DAP
                    Integer decrypted;
                    decrypted = paillier.decrypt(predicted_index);

                    mpf_class predicted_index_f, decrypted_f(decrypted), SCALED_f(SCALED), leaf_node_num_f(leaf_node_num);
                    predicted_index_f = (decrypted_f / (SCALED_f * SCALED_f * SCALED_f) * leaf_node_num_f);
                    decrypted = Integer(mpz_class(predicted_index_f));
                    predicted_index = paillier.encrypt(decrypted);
                    // DSP
                    denc_R *= Integer(leaf_node_num);
                    enc_R = paillier.encrypt(denc_R);
                    predicted_index = predicted_index - enc_R;
                    if (SICrun(predicted_index, enc_width, paillier) == 0)
                        predicted_index = enc_width;
                }
                else if(net_type == 1)
                {
                    predicted_index = netclass->enc_predict(vertexes[i], paillier);
                    //DAP
                    Integer denc_predicted_index = paillier.decrypt(predicted_index);
                    predicted_index = enc_class_to_z[denc_predicted_index.get_si()];

                }
                Ciphertext predicted_index_max, predicted_index_min;
                predicted_index_max = predicted_index + enc_max_error;
                predicted_index_min = predicted_index + enc_min_error;
                if(SICrun(min, predicted_index_min, paillier) == 0) min = predicted_index_min;
                if(SICrun(predicted_index_max, max, paillier) == 0) max = predicted_index_max;
            }
            if(SICrun(enc_0, min, paillier) == 0) front = enc_0;
            else front = min;
            if(SICrun(leafnodes_size, max, paillier) == 1) 
                back = leafnodes_size - enc_1;
            else back = max;
        }
        vector<LeafNodeNoiseRecord> noise_records(leafnodes.size());

        vector<LeafNode> move_leafnodes = leafnodes;
        add_noise_to_leafnodes(move_leafnodes, paillier, noise_records);

        Ciphertext i = front;
        // cout<<"front: "<<paillier.decrypt(front)<<" back: "<<paillier.decrypt(back)<<endl;

        while(SICrun(i, back, paillier) == 1)
        {
            // cout<<"i: "<<paillier.decrypt(i)<<endl;

            // DSP
            vector<Ciphertext> sub;
            for (int j = 0; j < enc_leafnodes_index.size(); j++)
            {
                sub.push_back(enc_leafnodes_index[j] - i);
            }
            // vector<LeafNodeNoiseRecord> noise_records;
            // add_noise_to_leafnodes(leafnodes, paillier, noise_records);
            // DAP
            Mbr mbr(dim);
            mbr.encrypt_mbr(paillier);
            // vector<Ciphertext> notice;
            LeafNode leafnode_1(mbr);
            LeafNodeNoiseRecord noise_record_1;
            int is_true = 0;
            for (int i = 0; i < move_leafnodes.size(); i++)
            {
                // cout<<"sub: "<<paillier.decrypt(sub[i])<<endl;
                // cout<<"i: "<<i<<", predicted_index: "<<paillier.decrypt(predicted_index)<<endl;
                if (paillier.decrypt(sub[i]) == 0)
                {
                    leafnode_1 = move_leafnodes[i];
                    noise_record_1 = noise_records[i];
                    is_true++;
                    break;
                    // notice.push_back(enc_1);
                }
                // else notice.push_back(enc_0);
            }
            // DSP
            remove_noise_from_leafnode(leafnode_1, paillier, noise_record_1);
            // cout<<"leafnodes[0].point: ";
            // leafnodes[0].mbr.enc_print(paillier);
            i = i + enc_1;
            if(!is_true) 
            {
                continue;
            }
            auto it_1 = leafnode_1.children->begin();
            Point point_1(dim);
            mpz_class size(leafnode_1.children->size());
            // size *= 2;
            vector<Point> points(size.get_si());
            size = 0;
            // cout<<101<<endl;
            while (it_1 != leafnode_1.children->end())
            {
                point_1 = *it_1;
                it_1++;
                points[size.get_si()] = point_1;

                size = size + 1;
                // point.print();
            }
            // if(leafnode_1.mbr.enc_interact(query_window, paillier) || leafnode_0.mbr.enc_interact(query_window, paillier))
            // if(leafnode_1.mbr.enc_interact(query_window, paillier))
            // if(leafnode_1.mbr.interact(query_window))
            if(leafnode_1.mbr.enc_interact(query_window, paillier))
            {
                exp_recorder.page_access += 1;
                // cout<<"103"<<endl;
                // cout<<"page_access: "<<exp_recorder.page_access<<endl;
                // #pragma omp parallel for
                for(int j = 0; j < size.get_si(); j++)
                {
                    // cout<<"j: "<<j<<endl;
                    Point point = points[j];
                    // if(query_window.enc_contains(point, paillier))
                    // if(query_window.contains(point))
                    if(query_window.enc_contains(point, paillier))
                    {
                        exp_recorder.Base_line_window_query_results.push_back(point);
                        // exp_recorder.Base_line_window_query_result_size++;
                    }
                }
            }
            // cout<<"i: "<<paillier.decrypt(i)<<endl;
            // cout<<"back: "<<paillier.decrypt(back)<<endl;
            // cout<<"flag: "<<SICrun(i, back, paillier)<<endl;
        }
        return;
    }
    else
    {
        int children_size = width;
        int front = children_size - 1;
        int back = 0;
        int predicted_index = 0;
        for (size_t i = 0; i < vertexes.size(); i++)
        {
            if(net_type == 0)
            {
                predicted_index = net->predict(vertexes[i]) * children_size;
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
            }
            else if(net_type == 1)
            {
                predicted_index = netclass->predict(vertexes[i]);
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
                predicted_index = static_cast<int>(class_to_z[predicted_index]);
            }
            if (predicted_index < front)
            {
                front = predicted_index;
            }
            if (predicted_index > back)
            {
                back = predicted_index;
            }
        }
        // cout<<"front_1: "<<front<<endl;
        // cout<<"back_1: "<<back<<endl;
        for (size_t i = front; i <= back; i++)
        {
            if (children.count(i) == 0)
            {
                continue;
            }
            // cout<<"i_mbr: "<<endl;
            // children[i].mbr.print();
            // cout<<"query_window_mbr: "<<endl;
            // query_window.print();
            if (children[i].mbr.interact(query_window))
            {
                children[i].Base_line_window_query(exp_recorder, vertexes, query_window, paillier);
            }
        }
    }
}


void RSMI::Secure_window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, PaillierFast& paillier)
{
    Ciphertext enc_1, enc_0;
    Integer one(1), zero(0);
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);
    // query_window.encrypt_mbr(paillier);
    if (SICrun(enc_1, enc_is_last, paillier) == 1)
    {
        Integer denc_leafnodes_size(leafnodes.size());
        Ciphertext leafnodes_size;
        leafnodes_size = paillier.encrypt(denc_leafnodes_size);
        Ciphertext front;
        front = leafnodes_size - enc_1;
        Ciphertext back = enc_0;
        Ciphertext enc_2;
        Integer two(2);
        enc_2 = paillier.encrypt(two);
        if(SICrun(enc_leaf_node_num, enc_0, paillier) == 1) return;
        else if(SICrun(enc_2, enc_leaf_node_num, paillier) == 0)
        {
            // cout<<2<<endl;
            front = enc_0;
            back = enc_0;
        }
        else 
        {
            // cout<<3<<endl;
            Ciphertext max = enc_0, min = enc_width;
            // #pragma omp parallel for
            for(size_t i = 0; i < vertexes.size(); i++)
            {
                Ciphertext predicted_index;
                if(net_type == 0)
                {
                    predicted_index = net->enc_predict(vertexes[i], paillier);

                    Integer flag;
                    flag = SICrun(enc_0, predicted_index, paillier);
                    if (flag == 0)
                        predicted_index = enc_0;

                    PublicKey pub = paillier.get_pub();
                    Integer Random, denc_R(generateRandomNumber()), n(pub.n);
                    Ciphertext enc_R;
                    Random = denc_R * (SCALED * SCALED * SCALED);
                    enc_R = paillier.encrypt(Random);
                    predicted_index = predicted_index + enc_R;

                    // DAP
                    Integer decrypted;
                    decrypted = paillier.decrypt(predicted_index);

                    mpf_class predicted_index_f, decrypted_f(decrypted), SCALED_f(SCALED), leaf_node_num_f(leaf_node_num);
                    predicted_index_f = (decrypted_f / (SCALED_f * SCALED_f * SCALED_f) * leaf_node_num_f);
                    decrypted = Integer(mpz_class(predicted_index_f));
                    predicted_index = paillier.encrypt(decrypted);
                    // DSP
                    denc_R *= Integer(leaf_node_num);
                    enc_R = paillier.encrypt(denc_R);
                    predicted_index = predicted_index - enc_R;
                    if (SICrun(predicted_index, enc_width, paillier) == 0)
                        predicted_index = enc_width;
                }
                else if(net_type == 1)
                {
                    predicted_index = netclass->enc_predict(vertexes[i], paillier);
                    //DAP
                    Integer denc_predicted_index = paillier.decrypt(predicted_index);
                    predicted_index = enc_class_to_z[denc_predicted_index.get_si()];
                    // if(SICrun(enc_0, predicted_index, paillier) == 0) predicted_index = enc_0;
                    // if(SICrun(predicted_index, enc_width, paillier) == 0) predicted_index = enc_width;
                }
                Ciphertext predicted_index_max, predicted_index_min;
                predicted_index_max = predicted_index + enc_max_error;
                predicted_index_min = predicted_index + enc_min_error;
                if(SICrun(min, predicted_index_min, paillier) == 0) min = predicted_index_min;
                if(SICrun(predicted_index_max, max, paillier) == 0) max = predicted_index_max;
            }
            if(SICrun(enc_0, min, paillier) == 0) front = enc_0;
            else front = min;
            if(SICrun(leafnodes_size, max, paillier) == 1) 
                back = leafnodes_size - enc_1;
            else back = max;
        }
        vector<LeafNodeNoiseRecord> noise_records(leafnodes.size());

        vector<LeafNode> move_leafnodes = leafnodes;
        add_noise_to_leafnodes(move_leafnodes, paillier, noise_records);

        Ciphertext i = front;
        // cout<<"front: "<<paillier.decrypt(front)<<" back: "<<paillier.decrypt(back)<<endl;

        while(SICrun(i, back, paillier) == 1)
        {
            // cout<<"i: "<<paillier.decrypt(i)<<endl;

            // DSP
            vector<Ciphertext> sub;
            for (int j = 0; j < enc_leafnodes_index.size(); j++)
            {
                sub.push_back(enc_leafnodes_index[j] - i);
            }

            // DAP
            Mbr mbr(dim);
            mbr.encrypt_mbr(paillier);
            // vector<Ciphertext> notice;
            LeafNode leafnode_1(mbr);
            LeafNodeNoiseRecord noise_record_1;
            int is_true = 0;
            for (int i = 0; i < move_leafnodes.size(); i++)
            {

                if (paillier.decrypt(sub[i]) == 0)
                {
                    leafnode_1 = move_leafnodes[i];
                    noise_record_1 = noise_records[i];
                    is_true++;
                    break;
                    // notice.push_back(enc_1);
                }
                // else notice.push_back(enc_0);
            }
            // DSP
            remove_noise_from_leafnode(leafnode_1, paillier, noise_record_1);

            i = i + enc_1;
            if(!is_true) 
            {
                continue;
            }
            auto it_1 = leafnode_1.children->begin();
            Point point_1(dim);
            mpz_class size(leafnode_1.children->size());
            // size *= 2;
            vector<Point> points(size.get_si());
            size = 0;
            // cout<<101<<endl;
            while (it_1 != leafnode_1.children->end())
            {
                point_1 = *it_1;
                it_1++;
                points[size.get_si()] = point_1;

                size = size + 1;
                // point.print();
            }

            if(leafnode_1.mbr.enc_interact(query_window, paillier))
            {
                exp_recorder.page_access += 1;
                // cout<<"page_access: "<<exp_recorder.page_access<<endl;
                // #pragma omp parallel for
                for(int j = 0; j < size.get_si(); j++)
                {
                    // cout<<"j: "<<j<<endl;
                    Point point = points[j];

                    if(query_window.enc_contains(point, paillier))
                    {
                        exp_recorder.Secure_window_query_results.push_back(point);
                        // exp_recorder.Base_line_window_query_result_size++;
                    }
                }
            }

        }
        return;
    }
    else
    {
 
        if(level > exp_recorder.depth - exp_recorder.enc_depth)
        {
            
            Ciphertext children_size = enc_width, front, back = enc_0;
            front = children_size - enc_1;
            for (size_t i = 0; i < vertexes.size(); i++)
            {
              
                Ciphertext predicted_index;
                if (net_type == 0)
                {
                    predicted_index = net->enc_predict(vertexes[i], paillier);
                    
                    Integer flag;
                    flag = SICrun(enc_0, predicted_index, paillier);
                    if (flag == 0)
                        predicted_index = enc_0;
                 
                    PublicKey pub = paillier.get_pub();
                    Integer Random, denc_R(generateRandomNumber()), n(pub.n);
                    Ciphertext enc_R;
                    Random = denc_R * (SCALED * SCALED * SCALED);
                    enc_R = paillier.encrypt(Random);
                    predicted_index = predicted_index + enc_R;
           ;
                    // DAP
                    Integer decrypted;
                    decrypted = paillier.decrypt(predicted_index);
                    if (decrypted > n / 2)
                    {
                        decrypted -= n; 
                    }
                   
                    Integer denc_children_size;
                    denc_children_size = paillier.decrypt(children_size);
                    mpf_class predicted_index_f, decrypted_f(decrypted), SCALED_f(SCALED), children_size_f(denc_children_size);
                    predicted_index_f = (decrypted_f / (SCALED_f * SCALED_f * SCALED_f) * children_size_f);
                    decrypted = Integer(mpz_class(predicted_index_f));
                    predicted_index = paillier.encrypt(decrypted);
                    
                    denc_R *= Integer(denc_children_size);
                    // DSP
                    enc_R = paillier.encrypt(denc_R);
                    predicted_index = predicted_index - enc_R;
                    if (SICrun(children_size, predicted_index, paillier) == 1) predicted_index = children_size - enc_1;
                }
                else if(net_type == 1)
                {
                    predicted_index = netclass->enc_predict(vertexes[i], paillier);
                    Integer denc_predicted_index = paillier.decrypt(predicted_index);
                    predicted_index = enc_class_to_z[denc_predicted_index.get_si()];
                }
     
                if (SICrun(front, predicted_index, paillier) == 0)
                    front = predicted_index;
                if (SICrun(predicted_index, back, paillier) == 0)
                    back = predicted_index;
            }
        
            Ciphertext l = front;
            RSMI move_children = *this;
            shared_ptr<RSMINoiseRecord> noise_rec = move_children.add_noise_to_all_children(paillier);
            while(SICrun(l, back, paillier) == 1)
            {
                // DSP
                vector<Ciphertext> sub;
                for (int i = 0; i < enc_children_index.size(); i++)
                {
                    sub.push_back(enc_children_index[i] - l);
                }
                l = l + enc_1;
                // shared_ptr<RSMINoiseRecord> noise_rec = this->add_noise_to_all_children(paillier);
                // DAP
                RSMI* children_1;
                std::map<int, std::shared_ptr<RSMINoiseRecord>>::const_iterator rit;
                // vector<Ciphertext> notice;
                // cout<<"enc_children_index.size: "<<enc_children_index.size()<<endl;
                int is_true = 0;
                for (int i = 0; i < enc_children_index.size(); i++)
                {

                    if (paillier.decrypt(sub[i]) == 0)
                    {
                        auto it = move_children.children.begin();
                        std::advance(it, i);
                        if (it == move_children.children.end())
                        {
                            // remove_noise_from_all_children(paillier, noise_rec);
                            continue;
                        }
                        children_1 = &it->second;
                        rit = noise_rec->child_noises.find(it->first);
                        is_true++;
                        break;
                        // notice.push_back(enc_1);
                    }
                    // else notice.push_back(enc_0);
                }
                // DSP

                if(!is_true) continue;
                remove_noise_from_rsmi_recursive(*children_1, paillier, rit->second);
                if (children_1->mbr.enc_interact(query_window, paillier))
                {
                    children_1->Secure_window_query(exp_recorder, vertexes, query_window, paillier);
                }
            }
            // return;
        }
        else
        {
            cout<<"nonleafnode"<<endl;
            // mbr.print();
            int children_size = width;
            int front = children_size - 1;
            int back = 0;
            int predicted_index = 0;
            for (size_t i = 0; i < vertexes.size(); i++)
            {
                if (net_type == 0)
                {
                    predicted_index = net->predict(vertexes[i]) * children_size;
                    predicted_index = predicted_index < 0 ? 0 : predicted_index;
                    predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
                }
                else if (net_type == 1)
                {
                    predicted_index = netclass->predict(vertexes[i]);
                    predicted_index = predicted_index < 0 ? 0 : predicted_index;
                    predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
                    predicted_index = static_cast<int>(class_to_z[predicted_index]);
                }
                if (predicted_index < front)
                {
                    front = predicted_index;
                }
                if (predicted_index > back)
                {
                    back = predicted_index;
                }
            }

            for (size_t i = front; i <= back; i++)
            {
                if (children.count(i) == 0)
                {
                    continue;
                }

                if (children[i].mbr.interact(query_window))
                {
                    children[i].Secure_window_query(exp_recorder, vertexes, query_window, paillier);
                }
            }
        }
    }
}


vector<Point> RSMI::acc_window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    // cout<<"acc_window_query:_level: "<<level<<endl;
    // mbr.print();
    vector<Point> window_query_results;
    if (is_last)
    {
        for (LeafNode leafnode : leafnodes)
        {
            if (leafnode.mbr.interact(query_window))
            {
                // leafnode.mbr.print();
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_window.contains(point))
                    {
                        window_query_results.push_back(point);
                    }
                }
            }
        }
    }
    else
    {
        map<int, RSMI>::iterator iter = children.begin();
        while (iter != children.end())
        {
            if (iter->second.mbr.interact(query_window))
            {
                vector<Point> tempResult = iter->second.acc_window_query(exp_recorder, query_window);
                window_query_results.insert(window_query_results.end(), tempResult.begin(), tempResult.end());
            }
            iter++;
        }
    }
    return window_query_results;
}

void RSMI::acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    int length = query_windows.size();

    for (int i = 0; i < length; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        query_windows[i].init();
        exp_recorder.acc_window_query_result_size += acc_window_query(exp_recorder, query_windows[i]).size();
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    cout<<"acc_window_query_qesult_size: "<<exp_recorder.acc_window_query_result_size<<endl;
    exp_recorder.time = exp_recorder.time / length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
}


// TODO when rebuild!!!
void RSMI::insert(ExpRecorder &exp_recorder, Point point, PaillierFast& paillier, string index_path)
{
    Constants constants;
    int page_size = exp_recorder.B; 
    Ciphertext enc_1, enc_0;
    Integer zero(0), one(1);
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);


    long long predicted_key = 0;
    if (is_last) {

        if (width <= 0) predicted_key = 0;
        else predicted_key = static_cast<long long>(net->predict(point) * width);
        if (predicted_key < 0) predicted_key = 0;
        if (width > 0 && predicted_key >= width) predicted_key = width - 1;
    } else {
       
        // if (net_type == 1 && netclass && class_to_z.size() > 0) {
        if (net_type == 1) {
            int predicted_class = static_cast<int>(netclass->predict(point));
            predicted_class = std::max(0, std::min((int)class_to_z.size() - 1, predicted_class));
        
            predicted_key = static_cast<long long>(class_to_z[predicted_class]);
        } else {
            if (width <= 0) predicted_key = 0;
            else predicted_key = static_cast<long long>(net->predict(point) * width);
            if (predicted_key < 0) predicted_key = 0;
            if (width > 0 && predicted_key >= width) predicted_key = width - 1;
        }
    }


    if (is_last) {

        if (N == exp_recorder.N) {
            cout << "rebuild: " << endl;
            is_last = false;
            enc_is_last = enc_0;

            vector<Point> points;
            for (LeafNode &leafNode : leafnodes) {
                points.insert(points.end(), leafNode.children->begin(), leafNode.children->end());
            }
            points.push_back(point); 

            // cout << "points.size: " << points.size() << endl;
            auto start = chrono::high_resolution_clock::now();
            kmeans_build(exp_recorder, points, paillier);
            auto finish = chrono::high_resolution_clock::now();
            exp_recorder.rebuild_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
            exp_recorder.rebuild_num++;
        }
        else {
    
            int insertedIndex = 0;
            if (page_size > 0)
                insertedIndex = static_cast<int>(predicted_key / page_size);
            
            if (insertedIndex < 0) insertedIndex = 0;
            if (insertedIndex >= leafnodes.size()) insertedIndex = (int)leafnodes.size() - 1;

            // split if full
            if (leafnodes[insertedIndex].is_full()) {
                LeafNode right = leafnodes[insertedIndex].split1();
                leafnodes.insert(leafnodes.begin() + insertedIndex + 1, right);
                leaf_node_num++;
                max_error++;
                min_error--;

                enc_leaf_node_num = enc_leaf_node_num + enc_1;
                enc_max_error = enc_max_error + enc_1;
                enc_min_error = enc_min_error - enc_1;
            }


            if (insertedIndex >= leafnodes.size()) insertedIndex = (int)leafnodes.size() - 1;


            leafnodes[insertedIndex].add_point(point);
            // leafnodes[insertedIndex].mbr.enc_update(point.enc_coords, paillier);
            leafnodes[insertedIndex].mbr.update(point);
            leafnodes[insertedIndex].mbr.encrypt_mbr(paillier);
            N++;
            width++;

            enc_N = enc_N + enc_1;
            enc_width = enc_width + enc_1;

            int front = insertedIndex + min_error;
            int back = insertedIndex + max_error;
            front = std::max(front, 0);
            int leafnodenum = (int)leafnodes.size() - 1;
            back = std::min(leafnodenum, back);

            if ((back - front) > (int)leafnodes.size() / 2)
            {
       
                vector<Point> points;
                for (LeafNode &leafNode : leafnodes)
                {
                    points.insert(points.end(), leafNode.children->begin(), leafNode.children->end());
                }
                N = points.size();

                int D = dim;
                if (N > 0)
                {
                    vector<vector<int>> ranks(D, vector<int>(N));
                    for (int d = 0; d < D; ++d)
                    {
                        vector<pair<float, int>> tmp;
                        tmp.reserve(N);
                        for (int i = 0; i < N; ++i)
                            tmp.emplace_back(points[i].coords[d], i);
                        sort(tmp.begin(), tmp.end());
                        for (int i = 0; i < N; ++i)
                            ranks[d][tmp[i].second] = i;
                    }
                    int bits = (N > 1) ? (int)ceil(log2((double)N)) : 1;
                    for (int i = 0; i < N; ++i)
                    {
                        std::vector<bitmask_t> coord(D);
                        for (int d = 0; d < D; ++d)
                            coord[d] = ranks[d][i];
                        points[i].curve_val = hilbert_c2i(D, bits, coord.data());
                    }
                    sort(points.begin(), points.end(), sort_curve_val());
                }

                if (N == 0)
                    width = 0;
                else if (N == 1)
                {
                    points[0].index = 0;
                    width = 0;
                }
                else
                {
                    width = N - 1;
                    for (int i = 0; i < N; ++i)
                    {
                        points[i].index = static_cast<float>(i) / (N - 1);
                    }
                }

                leafnodes.clear();
                leaf_node_num = (page_size > 0) ? (N / page_size) : 0;
                for (int i = 0; i < leaf_node_num; ++i)
                {
                    LeafNode leafNode(dim);
                    auto bn = points.begin() + i * page_size;
                    auto en = bn + page_size;
                    leafNode.add_points(vector<Point>(bn, en));
                    leafNode.mbr.encrypt_mbr(paillier);
                    leafnodes.push_back(leafNode);
                }
                if (page_size > 0 && (N % page_size != 0))
                {
                    LeafNode leafNode(dim);
                    auto bn = points.begin() + leaf_node_num * page_size;
                    leafNode.add_points(vector<Point>(bn, points.end()));
                    leafNode.mbr.encrypt_mbr(paillier);
                    leafnodes.push_back(leafNode);
                    leaf_node_num++;
                }
                else if (page_size == 0 && N > 0)
                {
                    LeafNode leafNode(dim);
                    leafNode.add_points(points);
                    leafNode.mbr.encrypt_mbr(paillier);
                    leafnodes.push_back(leafNode);
                    leaf_node_num = 1;
                }
                mbr.update(point);
                mbr.encrypt_mbr(paillier);

                net = std::make_shared<Net>(dim, leaf_node_num / 2 + 2);
                vector<float> locations;
                vector<float> labels;
                locations.reserve((size_t)N * dim);
                labels.reserve(N);
                for (Point &p : points)
                {
                    for (int d = 0; d < dim; ++d)
                        locations.push_back(p.coords[d]);
                    labels.push_back(p.index);
                }
                net->train_model(locations, labels);
                net->get_parameters();
                net->init_encrypted_params(paillier);

                max_error = 0;
                min_error = 0;
                for (int i = 0; i < N; ++i)
                {
                    Point &p = points[i];
                    int predicted = static_cast<int>(net->predict(p) * leaf_node_num);
                    predicted = std::max(0, std::min(predicted, leaf_node_num - 1));
                    int actual = i / ((page_size > 0) ? page_size : 1);
                    int error = actual - predicted;
                    max_error = std::max(max_error, error);
                    min_error = std::min(min_error, error);
                }


            }
            encrypt_node(paillier);
        }
    }

    else {
     
        if (children.count(predicted_key) == 0)
        {
           
            exp_recorder.result++;
            return;
        }
        children[predicted_key].insert(exp_recorder, point, paillier, index_path);
    }




void RSMI::insert(ExpRecorder &exp_recorder, vector<Point> points, PaillierFast& paillier, string index_path)
{
    auto start = chrono::high_resolution_clock::now();
    int i = 0;
    for (Point point : points)
    {
        cout<<"i: "<<i++<<endl;
        // point.print();
        insert(exp_recorder, point, paillier, index_path);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.insert_time = (chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.insert_num;
    cout<<"N: "<<N<<endl;
}
BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::binary_oarchive)
BOOST_CLASS_EXPORT_KEY(RSMI)