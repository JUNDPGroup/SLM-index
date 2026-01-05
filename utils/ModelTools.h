#ifndef MODELTOOLS_H
#define MODELTOOLS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
// 然后单独包含Torch核心头文件（避免重复包含）
// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 确保只包含一次 torch 头文件
#ifndef TORCH_ALREADY_INCLUDED
#define TORCH_ALREADY_INCLUDED
#include <torch/torch.h>
#endif

#include <gmp.h>
#include <ophelib/paillier_fast.h>
#include "../agreements/SIC.h"
#include "../agreements/SM.h"
#include "../agreements/SSED.h"
#include "./Constants.h"
#include "../Serialize/serialization_helpers.hpp"
#include <gmpxx.h>
#include <xmmintrin.h>
#include <chrono>
#include <thread>
#include <omp.h>
#include <xmmintrin.h>
#include <boost/filesystem.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
using namespace ophelib;
using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace std;
struct Net : torch::nn::Module
{
private:
    friend class boost::serialization::access;

    template <class Archive>
    void save(Archive &ar, const unsigned int version) const
    {
        ar & input_dim & max_error & min_error & width & learning_rate & b2 & SCALE;
        ar & normalize_zorder & z_min & z_max;
        ar & train_min_vals & train_max_vals;

        // 序列化权重数组
        ar & boost::serialization::make_array(w1, width * input_dim);
        ar & boost::serialization::make_array(w1_, width);
        ar & boost::serialization::make_array(w2, width);
        ar & boost::serialization::make_array(b1, width);

        for (int i = 0; i < 8; ++i)
        {
            ar &enc_w1_d[i];
        }
        ar & enc_b1;
        ar & enc_w2;
        ar & enc_b2;

        // 序列化SIMD数组
        for(int d=0; d<input_dim; d++) {
            ar & boost::serialization::make_array(w1_d[d], width);
        }
        ar & boost::serialization::make_array(w2_, width);
        ar & boost::serialization::make_array(b1_, width);
        ar & boost::serialization::make_array(w1__, width);
    }

    template <class Archive>
    void load(Archive &ar, const unsigned int version)
    {
        // 先释放内存
        // cleanup();
        
        // 加载基础参数
        ar & input_dim & max_error & min_error & width & learning_rate & b2 & SCALE;
        ar & normalize_zorder & z_min & z_max;
        ar & train_min_vals & train_max_vals;

        // 动态分配内存
        allocate_memory();

        // 反序列化权重数组
        ar & boost::serialization::make_array(w1, width * input_dim);
        ar & boost::serialization::make_array(w1_, width);
        ar & boost::serialization::make_array(w2, width);
        ar & boost::serialization::make_array(b1, width);

        for (int i = 0; i < 8; ++i)
        {
            ar &enc_w1_d[i];
        }
        ar & enc_b1;
        ar & enc_w2;
        ar & enc_b2;
        
        // 反序列化SIMD数组
        for(int d=0; d<input_dim; d++) {
            ar & boost::serialization::make_array(w1_d[d], width);
        }
        ar & boost::serialization::make_array(w2_, width);
        ar & boost::serialization::make_array(b1_, width);
        ar & boost::serialization::make_array(w1__, width);
    }

    // 清理内存的辅助函数
    void cleanup()
    {
        if (w1) _mm_free(w1);
        if (w1_) _mm_free(w1_);
        if (w2) _mm_free(w2);
        if (b1) _mm_free(b1);
        
        if (w1_d) {
            for (int d = 0; d < input_dim; d++) {
                if (w1_d[d]) _mm_free(w1_d[d]);
            }
            delete[] w1_d;
        }
        
        if (w2_) _mm_free(w2_);
        if (b1_) _mm_free(b1_);
        if (w1__) _mm_free(w1__);
        
        // 重置指针
        w1 = w1_ = w2 = b1 = w2_ = b1_ = w1__ = nullptr;
        w1_d = nullptr;
    }

    // 分配内存的辅助函数
    void allocate_memory()
    {
        w1 = (float *)_mm_malloc(width * input_dim * sizeof(float), 32);
        w1_ = (float *)_mm_malloc(width * sizeof(float), 32);
        w2 = (float *)_mm_malloc(width * sizeof(float), 32);
        b1 = (float *)_mm_malloc(width * sizeof(float), 32);
        
        w1_d = new float*[input_dim];
        for(int d=0; d<input_dim; d++) {
            w1_d[d] = (float *)_mm_malloc(width * sizeof(float), 32);
        }
        
        w2_ = (float *)_mm_malloc(width * sizeof(float), 32);
        b1_ = (float *)_mm_malloc(width * sizeof(float), 32);
        w1__ = (float *)_mm_malloc(width * sizeof(float), 32);
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    int input_dim;  // 输入维度（1-8维）
    int max_error = 0;
    int min_error = 0;
    int width = 0;
    float learning_rate = Constants::LEARNING_RATE;
    vector<float> train_min_vals;  // 记录各维度最小值
    vector<float> train_max_vals;  // 记录各维度最大值
    bool normalize_zorder = false;  // 是否对Z-Order归一化
    uint64_t z_min, z_max;           // Z-Order范围

    // 参数存储（原生数组）
    // float w1[Constants::HIDDEN_LAYER_WIDTH * 8]; // 支持最大8维
    // float w1_[Constants::HIDDEN_LAYER_WIDTH];
    // float w2[Constants::HIDDEN_LAYER_WIDTH];
    // float b1[Constants::HIDDEN_LAYER_WIDTH];

    // 修改w1数组为动态大小
    float *w1; // 改为动态分配，原：float w1[Constants::HIDDEN_LAYER_WIDTH * 8];
    float *w1_;
    float *w2;
    float *b1;

    // SIMD优化指针（每个维度一个指针）
    float **w1_d;  // 各维度权重指针
    float *w2_;
    float *b1_;
    float *w1__;
    float b2 = 0.0;

    // 加密参数（每个维度一个向量）
    std::vector<Ciphertext> enc_w1_d[8];
    std::vector<Ciphertext> enc_b1, enc_w2;
    Ciphertext enc_b2;
    
    std::chrono::milliseconds sic_time{0};
    std::chrono::milliseconds sm_time{0};
    long SCALE = 1e6;

    Net() : w1(nullptr), w1_(nullptr), w2(nullptr), b1(nullptr), w1_d(nullptr) {}

    // // === 深拷贝构造函数 ===
    // Net(const Net &other)
    // {
    //     // 基础字段
    //     input_dim = other.input_dim;
    //     max_error = other.max_error;
    //     min_error = other.min_error;
    //     width = other.width;
    //     learning_rate = other.learning_rate;
    //     train_min_vals = other.train_min_vals;
    //     train_max_vals = other.train_max_vals;
    //     normalize_zorder = other.normalize_zorder;
    //     z_min = other.z_min;
    //     z_max = other.z_max;
    //     SCALE = other.SCALE;
    //     sic_time = other.sic_time;
    //     sm_time = other.sm_time;
    //     b2 = other.b2;

    //     // ⚠️ 不再拷贝 fc1/fc2，保持默认 nullptr 即可

    //     // w1
    //     w1 = (other.w1) ? new float[width * input_dim] : nullptr;
    //     if (w1)
    //         std::copy(other.w1, other.w1 + width * input_dim, w1);

    //     w1_ = (other.w1_) ? new float[width] : nullptr;
    //     if (w1_)
    //         std::copy(other.w1_, other.w1_ + width, w1_);

    //     w2 = (other.w2) ? new float[width] : nullptr;
    //     if (w2)
    //         std::copy(other.w2, other.w2 + width, w2);

    //     b1 = (other.b1) ? new float[width] : nullptr;
    //     if (b1)
    //         std::copy(other.b1, other.b1 + width, b1);

    //     // w1_d
    //     if (other.w1_d)
    //     {
    //         w1_d = new float *[input_dim];
    //         for (int d = 0; d < input_dim; d++)
    //         {
    //             w1_d[d] = new float[width];
    //             std::copy(other.w1_d[d], other.w1_d[d] + width, w1_d[d]);
    //         }
    //     }
    //     else
    //     {
    //         w1_d = nullptr;
    //     }

    //     // w2_ / b1_ / w1__
    //     w2_ = (other.w2_) ? new float[width] : nullptr;
    //     if (w2_)
    //         std::copy(other.w2_, other.w2_ + width, w2_);

    //     b1_ = (other.b1_) ? new float[width] : nullptr;
    //     if (b1_)
    //         std::copy(other.b1_, other.b1_ + width, b1_);

    //     w1__ = (other.w1__) ? new float[width * input_dim] : nullptr;
    //     if (w1__)
    //         std::copy(other.w1__, other.w1__ + width * input_dim, w1__);

    //     // 加密权重
    //     for (int d = 0; d < 8; d++)
    //     {
    //         enc_w1_d[d] = other.enc_w1_d[d];
    //     }
    //     enc_b1 = other.enc_b1;
    //     enc_w2 = other.enc_w2;
    //     enc_b2 = other.enc_b2;
    // }

    // === 深拷贝构造函数 ===
    Net(const Net &other)
    {
        // 基础字段
        input_dim = other.input_dim;
        max_error = other.max_error;
        min_error = other.min_error;
        width = other.width;
        learning_rate = other.learning_rate;
        train_min_vals = other.train_min_vals;
        train_max_vals = other.train_max_vals;
        normalize_zorder = other.normalize_zorder;
        z_min = other.z_min;
        z_max = other.z_max;
        SCALE = other.SCALE;
        sic_time = other.sic_time;
        sm_time = other.sm_time;
        b2 = other.b2;

        // ⚠️ 显式置空 fc1/fc2
        fc1 = nullptr;
        fc2 = nullptr;

        // w1
        w1 = (other.w1) ? new float[width * input_dim] : nullptr;
        if (w1)
            std::copy(other.w1, other.w1 + width * input_dim, w1);

        w1_ = (other.w1_) ? new float[width] : nullptr;
        if (w1_)
            std::copy(other.w1_, other.w1_ + width, w1_);

        w2 = (other.w2) ? new float[width] : nullptr;
        if (w2)
            std::copy(other.w2, other.w2 + width, w2);

        b1 = (other.b1) ? new float[width] : nullptr;
        if (b1)
            std::copy(other.b1, other.b1 + width, b1);

        // w1_d
        if (other.w1_d)
        {
            w1_d = new float *[input_dim];
            for (int d = 0; d < input_dim; d++)
            {
                w1_d[d] = new float[width];
                std::copy(other.w1_d[d], other.w1_d[d] + width, w1_d[d]);
            }
        }
        else
        {
            w1_d = nullptr;
        }

        // w2_ / b1_ / w1__
        w2_ = (other.w2_) ? new float[width] : nullptr;
        if (w2_)
            std::copy(other.w2_, other.w2_ + width, w2_);

        b1_ = (other.b1_) ? new float[width] : nullptr;
        if (b1_)
            std::copy(other.b1_, other.b1_ + width, b1_);

        w1__ = (other.w1__) ? new float[width * input_dim] : nullptr;
        if (w1__)
            std::copy(other.w1__, other.w1__ + width * input_dim, w1__);

        // 加密权重
        for (int d = 0; d < 8; d++)
        {
            enc_w1_d[d] = other.enc_w1_d[d];
        }
        enc_b1 = other.enc_b1;
        enc_w2 = other.enc_w2;
        enc_b2 = other.enc_b2;
    }

    // // === 深拷贝赋值运算符 ===
    // Net& operator=(const Net &other) {
    //     if (this == &other) return *this;  // 避免自赋值

    //     // 先释放已有内存
    //     this->~Net();

    //     // 再用拷贝构造逻辑
    //     new (this) Net(other);

    //     return *this;
    // }


    ~Net()
    {
        // 添加析构函数释放内存
        if (w1) _mm_free(w1);
        if (w1_) _mm_free(w1_);
        if (w2) _mm_free(w2);
        if (b1) _mm_free(b1);
        if (w1_d) {
            for (int d = 0; d < input_dim; d++) {
                if (w1_d[d]) _mm_free(w1_d[d]);
            }
            delete[] w1_d;
        }
        if (w2_) _mm_free(w2_);
        if (b1_) _mm_free(b1_);
        if (w1__) _mm_free(w1__);
    }

    Net(int input_dim) : input_dim(input_dim)
    {
        this->width = Constants::HIDDEN_LAYER_WIDTH;
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, width));
        fc2 = register_module("fc2", torch::nn::Linear(width, 1));
        
        // 动态分配内存
        w1_d = new float*[input_dim];
        for(int d=0; d<input_dim; d++) {
            w1_d[d] = (float *)_mm_malloc(width * sizeof(float), 32);
        }
        w1 = (float *)_mm_malloc(width * input_dim * sizeof(float), 32);
        w1_ = (float *)_mm_malloc(width * sizeof(float), 32);
        w2 = (float *)_mm_malloc(width * sizeof(float), 32);
        b1 = (float *)_mm_malloc(width * sizeof(float), 32);
        w2_ = (float *)_mm_malloc(width * sizeof(float), 32);
        b1_ = (float *)_mm_malloc(width * sizeof(float), 32);
        w1__ = (float *)_mm_malloc(width * sizeof(float), 32);
        
        torch::nn::init::uniform_(fc1->weight, 0, 1);
        torch::nn::init::uniform_(fc2->weight, 0, 1);
    }

    Net(int input_dim, int width) : input_dim(input_dim)
    {
        this->width = min(width, Constants::HIDDEN_LAYER_WIDTH);
        
        // 动态分配内存（同上）
        w1_d = new float*[input_dim];
        for(int d=0; d<input_dim; d++) {
            w1_d[d] = (float *)_mm_malloc(this->width * sizeof(float), 32);
        }
        w1 = (float *)_mm_malloc(this->width * input_dim * sizeof(float), 32);
        w1_ = (float *)_mm_malloc(this->width * sizeof(float), 32);
        w2 = (float *)_mm_malloc(this->width * sizeof(float), 32);
        b1 = (float *)_mm_malloc(this->width * sizeof(float), 32);
        w2_ = (float *)_mm_malloc(this->width * sizeof(float), 32);
        b1_ = (float *)_mm_malloc(this->width * sizeof(float), 32);
        w1__ = (float *)_mm_malloc(this->width * sizeof(float), 32);
        
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, this->width));
        fc2 = register_module("fc2", torch::nn::Linear(this->width, 1));
        torch::nn::init::uniform_(fc1->weight, 0, 0.1);
        torch::nn::init::uniform_(fc2->weight, 0, 0.1);
    }

    // 模型保存方法（修改版）
    void save_model(const std::string &filepath) const
    {
        try
        {
            // 创建目录路径（如果不存在）
            boost::filesystem::path path(filepath);
            boost::filesystem::create_directories(path.parent_path());

            // 打开文件流
            std::ofstream ofs(filepath, std::ios::binary);
            if (!ofs.is_open())
            {
                throw std::runtime_error("Failed to open file for saving model: " + filepath);
            }

            // 使用Boost序列化模型
            boost::archive::binary_oarchive oa(ofs);
            oa << *this;
            ofs.close();
            std::cout << "Model saved successfully to: " << filepath << std::endl;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to save model: " + std::string(e.what()));
        }
    }

    // 模型加载方法（修改版）
    bool load_model(const std::string &filepath) // 修改为返回bool类型
    {
        try
        {
            // 检查文件是否存在
            if (!boost::filesystem::exists(filepath))
            {
                return false;
            }

            // 打开文件流
            std::ifstream ifs(filepath, std::ios::binary);
            if (!ifs.is_open())
            {
                return false;
            }

            // 使用Boost反序列化模型
            boost::archive::binary_iarchive ia(ifs);
            ia >> *this;
            ifs.close();
            std::cout << "Model loaded successfully from: " << filepath << std::endl;

            // 重建Torch模块（如果需要）
            // fc1 = register_module("fc1", torch::nn::Linear(input_dim, width));
            // fc2 = register_module("fc2", torch::nn::Linear(width, 1));

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Model loading error: " << e.what() << std::endl;
            return false;
        }
    }

    void get_parameters()
    {
        torch::Tensor p1 = this->parameters()[0];
        torch::Tensor p2 = this->parameters()[1];
        torch::Tensor p3 = this->parameters()[2];
        torch::Tensor p4 = this->parameters()[3];

        // 第一层权重
        p1 = p1.reshape({width, input_dim});
        for (size_t i = 0; i < width; i++) {
            for(int d=0; d<input_dim; d++) {
                w1[i * input_dim + d] = p1[i][d].item().toFloat();
                w1_d[d][i] = p1[i][d].item().toFloat();
                w1[i * input_dim + d] = (w1[i * input_dim + d] * SCALE) / SCALE;
                w1_d[d][i] = (w1_d[d][i] * SCALE) / SCALE;
                cout<<w1_d[d][i]<<" ";
            }
        }
        cout<<endl;
        // 第一层偏置
        p2 = p2.reshape({width, 1});
        for (size_t i = 0; i < width; i++) {
            b1[i] = p2.select(0, i).item().toFloat();
            b1_[i] = p2.select(0, i).item().toFloat();
            b1[i] = (b1[i] * SCALE) / SCALE;
            b1_[i] = (b1_[i] * SCALE) / SCALE;
            cout<<b1[i]<<" ";
        }
        // cout<<endl;

        // 第二层权重和偏置
        p3 = p3.reshape({width, 1});
        for (size_t i = 0; i < width; i++) {
            w2[i] = p3.select(0, i).item().toFloat();
            w2_[i] = p3.select(0, i).item().toFloat();
            w2[i] = (w2[i] * SCALE) / SCALE;
            w2_[i] = (w2_[i] * SCALE) / SCALE;
            cout<<w2[i]<<" ";
        }
        cout<<endl;
        b2 = p4.item().toFloat();
        b2 = (b2 * SCALE) / SCALE;
        cout<<b2<<endl;
        cout<<endl;
    }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
    
    torch::Tensor predict(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    // // 多维数据预测函数（安全版本）
    // float predict(const Point &point) const
    // {
    //     vector<float> coords = std::move(point.coords);
    //     // 检查输入点维度
    //     if (coords.size() != static_cast<size_t>(input_dim)) {
    //         throw std::invalid_argument("Point dimension does not match network input dimension");
    //     }
        
    //     constexpr int SIMD_WIDTH = 4;
    //     const int full_blocks = width / SIMD_WIDTH;
    //     const int rem = width % SIMD_WIDTH;
    //     const int total_width = width;  // 避免重复计算

    //     float result = 0.0f;
    //     // double result = 0.0f;
        
    //     // 处理完整的SIMD块
    //     for (int block = 0; block < full_blocks; block++) {
    //         const int offset = block * SIMD_WIDTH;
            
    //         // 使用对齐内存加载权重
    //         alignas(16) float block_result = 0.0f;
            
    //         // 处理每个SIMD组内的4个神经元
    //         for (int j = 0; j < SIMD_WIDTH; j++) {
    //             const int neuron_idx = offset + j;
                
    //             // 计算单个神经元的加权和
    //             float sum = 0.0f;
    //             for (int d = 0; d < input_dim; d++) {
    //                 sum += coords[d] * w1_d[d][neuron_idx];
    //             }
    //             sum += b1[neuron_idx];
                
    //             // ReLU激活
    //             if (sum > 0.0f) {
    //                 block_result += sum * w2[neuron_idx];
    //             }
    //         }
            
    //         result += block_result;
    //     }
        
    //     // 处理剩余神经元
    //     const int rem_start = full_blocks * SIMD_WIDTH;
    //     for (int i = rem_start; i < rem_start + rem; i++) {
    //         float sum = 0.0f;
    //         for (int d = 0; d < input_dim; d++) {
    //             sum += coords[d] * w1_d[d][i];
    //         }
    //         sum += b1[i];
            
    //         if (sum > 0.0f) {
    //             result += sum * w2[i];
    //         }
    //     }

    //     // 加上最终偏置
    //     result += b2;
    //     return result;
    // }

    double predict(const Point &point) const
    {
        auto int_param = [&](float val, int scale_factor) -> Integer {
            mpf_class SCALED(SCALE), VAL(val);
            for(int i = 0; i < scale_factor; i++) {
                VAL *= SCALED;
            }
            
            Integer scaled = mpz_class(VAL);
            // cout<<"scaled: "<<scaled<<endl;
            return scaled;
        };
        const std::vector<float> coords = std::move(point.coords);

        if (coords.size() != static_cast<size_t>(input_dim))
        {
            cout<<"dim: "<<point.dim<<" "<<input_dim<<endl;
            throw std::invalid_argument("Point dimension does not match network input dimension");
        }

        // 第1层：隐藏层计算（带ReLU）
        std::vector<Integer> hidden(width);

        for (int i = 0; i < width; ++i)
        {
            Integer sum = 0;

            for (int d = 0; d < input_dim; ++d)
            {
                Integer coord_int = int_param(coords[d], 1);
                // Integer enc_coord_int = paillier.decrypt(point.enc_coords[d]);
                // cout<<"denc_coords: "<<coord_int<<"enc_coords: "<<enc_coord_int<<endl;
                Integer weight_int = int_param(w1_d[d][i], 1);
                // Integer enc_weight_int = paillier.decrypt(enc_w1_d[d][i]);
                // cout<<"denc_wight1: "<<weight_int<<"enc_wight1: "<<enc_weight_int<<endl;
                sum += Integer(coord_int) * Integer(weight_int); // SCALE^2
            }

            Integer bias1_int = int_param(b1[i], 2);
            // Integer enc_bias1_int = paillier.decrypt(enc_b1[i]);
            // cout<<"denc_bias1: "<<bias1_int<<"enc_bias1: "<<enc_bias1_int<<endl;
            hidden[i] = sum + Integer(bias1_int); // SCALE^2（统一单位）

            // ReLU 激活函数
            if (hidden[i] < 0)
                hidden[i] = 0;

            // hidden[i] = sum; // 单位仍为 SCALE^2
        }

        // 第2层输出层计算
        Integer result = 0;
        for (int i = 0; i < width; ++i)
        {
            Integer weight2_int = int_param(w2[i], 1);
            // Integer enc_weight2_int = paillier.decrypt(enc_w2[i]);
            // cout<<"denc_wight2: "<<weight2_int<<"enc_wight2: "<<enc_weight2_int<<endl;
            result = result + hidden[i] * Integer(weight2_int); // hidden(SCALE^2) * weight2(SCALE) = SCALE^3
        }

        Integer bias2_int = int_param(b2, 3);
        // Integer enc_bias2_int = paillier.decrypt(enc_b2);
        // cout<<"denc_bias2: "<<bias2_int<<"enc_bias2: "<<enc_bias2_int<<endl;
        result = result + Integer(bias2_int); // 同样补足为 SCALE^3
        // cout<<"result: "<<result<<endl;
        mpf_class res_f(result);
        res_f /= (SCALE * SCALE * SCALE); // 等价于 result / 1e18（更高精度）
        return res_f.get_d();             // 转为 double（仍然最终是 float）
    }

    void init_encrypted_params(PaillierFast& paillier) {
        // 初始化加密参数存储
        for(int d=0; d<input_dim; d++) {
            enc_w1_d[d] = std::vector<Ciphertext>(width);
        }
        enc_b1 = std::vector<Ciphertext>(width);
        enc_w2 = std::vector<Ciphertext>(width);
        
        PublicKey pub = paillier.get_pub();
        Integer n(pub.n);
        
        auto encrypt_param = [&](float val, int scale_factor) -> Ciphertext {
            mpf_class SCALED(SCALE), VAL(val);
            for(int i = 0; i < scale_factor; i++) {
                VAL *= SCALED;
            }
            
            Integer scaled = mpz_class(VAL);
            return paillier.encrypt(scaled);
        };
    
        // 加密各维度权重
        for(int d=0; d<input_dim; d++) {
            for(int i=0; i<width; ++i) {
                enc_w1_d[d][i] = encrypt_param(w1_d[d][i], 1);
            }
        }
    
        // 加密偏置和输出权重
        for(int i=0; i<width; ++i) {
            enc_b1[i] = encrypt_param(b1[i], 2);
            enc_w2[i] = encrypt_param(w2[i], 1);
        }
    
        // 加密最终偏置
        enc_b2 = encrypt_param(b2, 3);
    }
    
    Ciphertext enc_predict(const Point &point, PaillierFast &paillier)
    {
        // 1. 加密预测（保持原有逻辑）
        vector<Ciphertext> enc_coods = std::move(point.enc_coords);
        std::vector<Ciphertext> hidden(width);
        Integer zero(0);
        Ciphertext enc_0 = paillier.encrypt(zero);
        Ciphertext output = enc_0;

        // 第一层计算
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < width; ++i)
        {
            // int tid = omp_get_thread_num();
            // #pragma omp critical
            // {
            //     std::cout << "Iteration " << i << " executed by thread " << tid << std::endl;
            // }
            Ciphertext sum = enc_0;
            for (int d = 0; d < input_dim; d++)
            {
                Ciphertext product = SMrun(enc_coods[d], enc_w1_d[d][i], paillier);
                sum = sum + product;
            }
            hidden[i] = sum + enc_b1[i];
            secure_relu(hidden[i], hidden[i], paillier);
        }

        // 输出层计算
        for (int i = 0; i < width; ++i)
        {
            Ciphertext product = SMrun(hidden[i], enc_w2[i], paillier);
            output = output + product;
        }
        output = output + enc_b2;
        return output;
    }

    void secure_relu(Ciphertext& res, Ciphertext& cipher, PaillierFast& paillier) {
        Integer zero(0), one(1);
        Ciphertext enc_0 = paillier.encrypt(zero);
        Ciphertext enc_1 = paillier.encrypt(one);
        
        Integer cmp_result = SICrun(enc_0, cipher, paillier);
        res = (cmp_result == 0) ? enc_0 : cipher;
        // Integer a = paillier.decrypt(cipher);
        // cout<<"a: "<<a<<endl;
        // if(a < 0) res = enc_0;
        // else res = cipher;
    }

    float activation(float val) {
        return (val > 0.0) ? val : 0.0;
    }

    /**
     * @brief 将多维数据点和Z-Order值转换为训练所需的locations和labels
     *
     * @param points 多维数据点集合，每个点是一个vector<float>表示各维度值
     * @param z_values 对应的Z-Order值集合
     * @param locations [输出] 展平后的训练数据位置(每input_dim个元素组成一个点)
     * @param labels [输出] 对应的Z-Order值标签
     * @param normalize 是否进行归一化处理(默认true)
     * @param min_vals 各维度最小值(用于归一化，可选)
     * @param max_vals 各维度最大值(用于归一化，可选)
     */
    // void prepare_training_data(
    //     const vector<vector<float>> &points,
    //     const vector<uint64_t> &z_values,
    //     vector<float> &locations,
    //     vector<float> &labels,
    //     bool normalize = true,
    //     const vector<float> &min_vals = {},
    //     const vector<float> &max_vals = {})
    // {
    //     // 检查输入有效性
    //     if (points.empty() || points.size() != z_values.size())
    //     {
    //         throw std::invalid_argument("Points and z_values must have same size and not empty");
    //     }

    //     const size_t num_points = points.size();
    //     const size_t input_dim = points[0].size();

    //     // 检查所有点维度一致
    //     for (const auto &point : points)
    //     {
    //         if (point.size() != input_dim)
    //         {
    //             throw std::invalid_argument("All points must have same dimension");
    //         }
    //     }

    //     // 准备归一化参数
    //     vector<float> actual_min_vals(input_dim, FLT_MAX);
    //     vector<float> actual_max_vals(input_dim, FLT_MIN);

    //     if (normalize)
    //     {
    //         // 如果未提供min/max值，则自动计算
    //         if (min_vals.empty() || max_vals.empty())
    //         {
    //             for (size_t i = 0; i < num_points; ++i)
    //             {
    //                 for (size_t d = 0; d < input_dim; ++d)
    //                 {
    //                     actual_min_vals[d] = std::min(actual_min_vals[d], points[i][d]);
    //                     actual_max_vals[d] = std::max(actual_max_vals[d], points[i][d]);
    //                 }
    //             }
    //         }
    //         else
    //         {
    //             // 使用提供的min/max值
    //             if (min_vals.size() != input_dim || max_vals.size() != input_dim)
    //             {
    //                 throw std::invalid_argument("Provided min_vals/max_vals must match input dimension");
    //             }
    //             actual_min_vals = min_vals;
    //             actual_max_vals = max_vals;
    //         }
    //     }

    //     // 准备输出容器
    //     locations.clear();
    //     locations.reserve(num_points * input_dim);
    //     vector<uint64_t> local_z_keys(std::move(z_values));
    //     // labels.assign(z_values.begin(), z_values.end());
    //     labels.clear();

    //     // 转换数据
    //     for (size_t i = 0; i < num_points; ++i)
    //     {
    //         for (size_t d = 0; d < input_dim; ++d)
    //         {
    //             float val = points[i][d];

    //             // 归一化处理
    //             if (normalize)
    //             {
    //                 float range = actual_max_vals[d] - actual_min_vals[d];
    //                 if (range > 0)
    //                 {
    //                     val = (val - actual_min_vals[d]) / range;
    //                 }
    //                 else
    //                 {
    //                     val = 0.5f; // 所有值相同的情况
    //                 }
    //             }

    //             locations.push_back(val);
    //         }
    //     }

    //     // 保存归一化参数
    //     if (normalize)
    //     {
    //         this->train_min_vals = actual_min_vals;
    //         this->train_max_vals = actual_max_vals;

    //         // 可选：记录Z-Order范围
    //         auto [min_it, max_it] = std::minmax_element(z_values.begin(), z_values.end());
    //         this->z_min = *min_it;
    //         this->z_max = *max_it;
    //         this->normalize_zorder = true;
    //     }
    //     // 对Z-Order值也进行归一化(可选)
    //     if (normalize)
    //     {
    //         auto minmax = std::minmax_element(local_z_keys.begin(), local_z_keys.end());
    //         uint64_t min_z = *minmax.first;
    //         uint64_t max_z = *minmax.second;
    //         uint64_t range_z = max_z - min_z;

    //         if (range_z > 0)
    //         {
    //             for (auto &z : local_z_keys)
    //             {
    //                 float z_f;
    //                 z_f= float((z - min_z) / (range_z));
    //                 // cout<<"归一化z值: "<<z_f<<endl;
    //                 labels.push_back(z_f);
    //             }
    //         }
    //     }
    //     // delete &local_z_keys;
    // }

    // void train_model(const vector<float> &locations, const vector<float> &labels)
    // {
    //     const int N = labels.size();
    //     const int input_dim = this->input_dim;

    //     // 一次性创建整个数据集张量（避免重复创建）
    //     torch::Tensor x_full = torch::from_blob(const_cast<float *>(locations.data()),
    //                                             {N, input_dim}, torch::kFloat)
    //                                .clone();
    //     torch::Tensor y_full = torch::from_blob(const_cast<float *>(labels.data()),
    //                                             {N, 1}, torch::kFloat)
    //                                .clone();

    //     // 添加requires_grad属性
    //     x_full.set_requires_grad(true);
    //     y_full.set_requires_grad(true);

    //     const int batch_size = std::min(4096, N); // 动态批量大小
    //     const int num_batches = (N + batch_size - 1) / batch_size;

    //     // cout << "Training size: " << N << " points, " << input_dim << " dimensions" << endl;
    //     // cout << "Using batch size: " << batch_size << ", total batches: " << num_batches << endl;

    //     torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learning_rate));

    //     for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
    //     {
    //         float epoch_loss = 0.0;
    //         for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
    //         {
    //             // 计算当前批次的范围
    //             const int start_idx = batch_idx * batch_size;
    //             const int end_idx = std::min(start_idx + batch_size, N);
    //             const int current_batch_size = end_idx - start_idx;

    //             // 创建批次切片（高效，不复制数据）
    //             torch::Tensor x_batch = x_full.slice(0, start_idx, end_idx).detach();
    //             torch::Tensor y_batch = y_full.slice(0, start_idx, end_idx).detach();

    //             optimizer.zero_grad();
    //             torch::Tensor predictions = forward(x_batch);
    //             torch::Tensor loss = torch::mse_loss(predictions, y_batch);
    //             epoch_loss += loss.item<float>() * current_batch_size;

    //             loss.backward();
    //             optimizer.step();
    //         }

    //         epoch_loss /= N;
    //         // cout << "Epoch: " << epoch + 1 << "/" << Constants::EPOCH
    //         //      << ", Avg Loss: " << epoch_loss << endl;
    //     }
    //     // cout << "Training completed" << endl;
    // }

    // void train_model(const vector<float> &locations, const vector<float> &labels)
    // {
    //     const int N = labels.size();
    //     const int input_dim = this->input_dim;

    //     torch::Tensor x_full = torch::from_blob(const_cast<float *>(locations.data()),
    //                                             {N, input_dim}, torch::kFloat)
    //                                .clone();
    //     torch::Tensor y_full = torch::from_blob(const_cast<float *>(labels.data()),
    //                                             {N, 1}, torch::kFloat)
    //                                .clone();

    //     const int batch_size = std::min(4096, N);
    //     const int num_batches = (N + batch_size - 1) / batch_size;

    //     torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learning_rate));

    //     float prev_loss = std::numeric_limits<float>::max();
    //     const float tol = 1e-3f; // 损失变化阈值
    //     const int patience = 100; // 容忍连续多少个epoch没有改进
    //     int patience_counter = 0;

    //     // for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
    //     for (size_t epoch = 0; ; epoch++)
    //     {
    //         float epoch_loss = 0.0;

    //         for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
    //         {
    //             int start_idx = batch_idx * batch_size;
    //             int end_idx = std::min(start_idx + batch_size, N);

    //             torch::Tensor x_batch = x_full.slice(0, start_idx, end_idx);
    //             torch::Tensor y_batch = y_full.slice(0, start_idx, end_idx);

    //             optimizer.zero_grad();
    //             torch::Tensor predictions = forward(x_batch);
    //             torch::Tensor loss = torch::mse_loss(predictions, y_batch);
    //             epoch_loss += loss.item<float>() * (end_idx - start_idx);

    //             loss.backward();
    //             optimizer.step();
    //         }

    //         epoch_loss /= N;

    //         if(epoch_loss <= tol) 
    //         {
    //             std::cout << "Loss near zero, stopping at epoch " << epoch
    //                 << " with loss = " << epoch_loss << std::endl;
    //             return;
    //         }
    //         // // 判断是否收敛
    //         // if (std::abs(prev_loss - epoch_loss) < tol)
    //         // {
    //         //     patience_counter++;
    //         //     if (patience_counter >= patience)
    //         //     {
    //         //         std::cout << "Early stopping at epoch " << epoch + 1
    //         //                   << " with loss = " << epoch_loss << std::endl;
    //         //         break;
    //         //     }
    //         // }
    //         // else
    //         // {
    //         //     patience_counter = 0; // 重置计数
    //         // }

    //         // prev_loss = epoch_loss;
    //     }
    // }

//     void train_model(std::vector<float> locations, std::vector<float> labels)
//     {
//         double target_loss = 1e-6;      // 目标损失
//         int patience = 10;              // 容忍无提升次数
//         double min_delta_ratio = 0.001; // 相对改善幅度阈值(0.1%)
//         long long N = labels.size();

//         // ====== Step 1: 打乱数据 ======
//         std::vector<long long> indices(N);
//         std::iota(indices.begin(), indices.end(), 0);
//         std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

//         std::vector<float> shuffled_locations;
//         std::vector<float> shuffled_labels;
//         shuffled_locations.reserve(locations.size());
//         shuffled_labels.reserve(labels.size());

//         for (auto idx : indices)
//         {
//             shuffled_locations.insert(shuffled_locations.end(),
//                                       locations.begin() + idx * this->input_dim,
//                                       locations.begin() + (idx + 1) * this->input_dim);
//             shuffled_labels.push_back(labels[idx]);
//         }

// #ifdef use_gpu
//         torch::Tensor x = torch::from_blob(shuffled_locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
//         torch::Tensor y = torch::from_blob(shuffled_labels.data(),
//                                            {N, 1},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
// #else
//         torch::Tensor x = torch::from_blob(shuffled_locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
//         torch::Tensor y = torch::from_blob(shuffled_labels.data(),
//                                            {N, 1},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
// #endif

//         // ====== Step 2: 初始化优化器 ======
//         float learn_rate = this->learning_rate;
//         torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));

//         double best_loss = std::numeric_limits<double>::max();
//         int no_improve_count = 0;

//         auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
//         {
//             optimizer.zero_grad();
//             auto loss = torch::mse_loss(this->forward(xb), yb);
//             loss.backward();
//             optimizer.step();
//             return loss.item<double>();
//         };

//         // ====== Step 3: 训练循环 ======
//         if (N > 64000000) // 大数据集模式
//         {
//             size_t batch_num = 4;
//             bool training_done = false;

//             while (!training_done)
//             {
//                 try
//                 {
//                     auto x_chunks = x.chunk(batch_num, 0);
//                     auto y_chunks = y.chunk(batch_num, 0);

//                     while (true)
//                     {
//                         double epoch_loss = 0.0;
//                         for (size_t i = 0; i < batch_num; i++)
//                         {
//                             epoch_loss += train_step(x_chunks[i], y_chunks[i]);
//                         }
//                         epoch_loss /= batch_num;

//                         std::cout << "Loss: " << epoch_loss << " - LR: " << learn_rate << std::endl;

//                         // ===== 改进后的相对改善判断 =====
//                         double improvement_ratio = (best_loss - epoch_loss) / best_loss;
//                         if (improvement_ratio > min_delta_ratio)
//                         {
//                             best_loss = epoch_loss;
//                             no_improve_count = 0;
//                         }
//                         else
//                         {
//                             no_improve_count++;
//                             if (no_improve_count >= patience)
//                             {
//                                 if (learn_rate > 1e-8)
//                                 {
//                                     learn_rate *= 0.5;
//                                     optimizer = torch::optim::Adam(this->parameters(),
//                                                                    torch::optim::AdamOptions(learn_rate));
//                                     std::cout << "学习率下降到 " << learn_rate << std::endl;
//                                     no_improve_count = 0;
//                                 }
//                                 else
//                                 {
//                                     std::cout << "早停触发，结束训练。" << std::endl;
//                                     training_done = true;
//                                     // break;
//                                     return;
//                                 }
//                             }
//                         }

//                         if (epoch_loss <= target_loss)
//                         {
//                             std::cout << "达到目标损失值，结束训练。" << std::endl;
//                             training_done = true;
//                             // break;
//                             return;
//                         }
//                     }
//                 }
//                 catch (const c10::Error &e)
//                 {
//                     std::cerr << "显存不足，batch_num 从 " << batch_num
//                               << " 增加到 " << batch_num * 2 << std::endl;
//                     batch_num *= 2;
//                 }
//             }
//         }
//         else // 小数据集模式
//         {
//             while (true)
//             {
//                 double loss_value = train_step(x, y);
//                 // std::cout << "Loss: " << loss_value << " - LR: " << learn_rate << std::endl;

//                 // ===== 改进后的相对改善判断 =====
//                 double improvement_ratio = (best_loss - loss_value) / best_loss;
//                 if (improvement_ratio > min_delta_ratio)
//                 {
//                     best_loss = loss_value;
//                     no_improve_count = 0;
//                 }
//                 else
//                 {
//                     no_improve_count++;
//                     if (no_improve_count >= patience)
//                     {
//                         if (learn_rate > 1e-8)
//                         {
//                             learn_rate *= 0.5;
//                             optimizer = torch::optim::Adam(this->parameters(),
//                                                            torch::optim::AdamOptions(learn_rate));
//                             std::cout << "学习率下降到 " << learn_rate << std::endl;
//                             no_improve_count = 0;
//                         }
//                         else
//                         {
//                             std::cout << "早停触发，结束训练。" << std::endl;
//                             // break;
//                             return;
//                         }
//                     }
//                 }

//                 if (loss_value <= target_loss)
//                 {
//                     std::cout << "达到目标损失值，结束训练。" << std::endl;
//                     // break;
//                     return;
//                 }
//             }
//         }
//     }

//     void train_model(std::vector<float> locations, std::vector<float> labels, double loss)
//     {
//         double target_loss = loss;      // 更严格的目标损失
//         int patience = 50;               // 容忍无提升次数（更大，减少早停可能）
//         double min_delta_ratio = 0.0001; // 相对改善幅度阈值(0.01%)
//         long long N = labels.size();

//         // ====== Step 1: 打乱数据 ======
//         std::vector<long long> indices(N);
//         std::iota(indices.begin(), indices.end(), 0);
//         std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

//         std::vector<float> shuffled_locations;
//         std::vector<float> shuffled_labels;
//         shuffled_locations.reserve(locations.size());
//         shuffled_labels.reserve(labels.size());

//         for (auto idx : indices)
//         {
//             shuffled_locations.insert(shuffled_locations.end(),
//                                       locations.begin() + idx * this->input_dim,
//                                       locations.begin() + (idx + 1) * this->input_dim);
//             shuffled_labels.push_back(labels[idx]);
//         }

// #ifdef use_gpu
//         torch::Tensor x = torch::from_blob(shuffled_locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
//         torch::Tensor y = torch::from_blob(shuffled_labels.data(),
//                                            {N, 1},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
// #else
//         torch::Tensor x = torch::from_blob(shuffled_locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
//         torch::Tensor y = torch::from_blob(shuffled_labels.data(),
//                                            {N, 1},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
// #endif

//         // ====== Step 2: 初始化优化器 ======
//         float learn_rate = this->learning_rate;
//         torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));

//         double best_loss = std::numeric_limits<double>::max();
//         int no_improve_count = 0;

//         auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
//         {
//             optimizer.zero_grad();
//             auto loss = torch::mse_loss(this->forward(xb), yb);
//             loss.backward();
//             optimizer.step();
//             return loss.item<double>();
//         };

//         // ====== Step 3: 训练循环 ======
//         if (N > 64000000) // 大数据集模式
//         {
//             size_t batch_num = 4;
//             bool training_done = false;

//             while (!training_done)
//             {
//                 try
//                 {
//                     auto x_chunks = x.chunk(batch_num, 0);
//                     auto y_chunks = y.chunk(batch_num, 0);

//                     while (true)
//                     {
//                         double epoch_loss = 0.0;
//                         for (size_t i = 0; i < batch_num; i++)
//                         {
//                             epoch_loss += train_step(x_chunks[i], y_chunks[i]);
//                         }
//                         epoch_loss /= batch_num;

//                         std::cout << "Loss: " << epoch_loss << " - LR: " << learn_rate << std::endl;

//                         double improvement_ratio = (best_loss - epoch_loss) / best_loss;
//                         if (improvement_ratio > min_delta_ratio)
//                         {
//                             best_loss = epoch_loss;
//                             no_improve_count = 0;
//                         }
//                         else
//                         {
//                             no_improve_count++;
//                             if (no_improve_count >= patience)
//                             {
//                                 if (learn_rate > 1e-12)
//                                 {
//                                     learn_rate *= 0.5;
//                                     optimizer = torch::optim::Adam(this->parameters(),
//                                                                    torch::optim::AdamOptions(learn_rate));
//                                     std::cout << "学习率下降到 " << learn_rate << std::endl;
//                                     no_improve_count = 0;
//                                 }
//                                 else
//                                 {
//                                     std::cout << "早停触发，结束训练。" << std::endl;
//                                     training_done = true;
//                                     return;
//                                 }
//                             }
//                         }

//                         if (epoch_loss <= target_loss)
//                         {
//                             std::cout << "达到目标损失值，结束训练。" << std::endl;
//                             training_done = true;
//                             return;
//                         }
//                     }
//                 }
//                 catch (const c10::Error &e)
//                 {
//                     std::cerr << "显存不足，batch_num 从 " << batch_num
//                               << " 增加到 " << batch_num * 2 << std::endl;
//                     batch_num *= 2;
//                 }
//             }
//         }
//         else // 小数据集模式
//         {
//             while (true)
//             {
//                 double loss_value = train_step(x, y);
//                 std::cout << "Loss: " << loss_value << " - LR: " << learn_rate << std::endl;

//                 double improvement_ratio = (best_loss - loss_value) / best_loss;
//                 if (improvement_ratio > min_delta_ratio)
//                 {
//                     best_loss = loss_value;
//                     no_improve_count = 0;
//                 }
//                 else
//                 {
//                     no_improve_count++;
//                     if (no_improve_count >= patience)
//                     {
//                         if (learn_rate > 1e-12)
//                         {
//                             learn_rate *= 0.5;
//                             optimizer = torch::optim::Adam(this->parameters(),
//                                                            torch::optim::AdamOptions(learn_rate));
//                             std::cout << "学习率下降到 " << learn_rate << std::endl;
//                             no_improve_count = 0;
//                         }
//                         else
//                         {
//                             std::cout << "早停触发，结束训练。" << std::endl;
//                             return;
//                         }
//                     }
//                 }

//                 if (loss_value <= target_loss)
//                 {
//                     std::cout << "达到目标损失值，结束训练。" << std::endl;
//                     return;
//                 }
//             }
//         }
//     }

    void train_model(std::vector<float> locations, std::vector<float> labels)
    {
        double target_loss = 1e-1;       // 目标损失
        int patience = 50;               // 容忍无提升次数
        double min_delta_ratio = 0.0001; // 相对改善幅度阈值
        long long N = labels.size();

        // ====== Step 1: 打乱数据 ======
        std::vector<long long> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 gen(42);
        // std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<float> shuffled_locations;
        std::vector<float> shuffled_labels;
        shuffled_locations.reserve(locations.size());
        shuffled_labels.reserve(labels.size());

        for (auto idx : indices)
        {
            shuffled_locations.insert(shuffled_locations.end(),
                                      locations.begin() + idx * this->input_dim,
                                      locations.begin() + (idx + 1) * this->input_dim);
            shuffled_labels.push_back(labels[idx]);
        }

#ifdef use_gpu
        torch::Tensor x = torch::from_blob(shuffled_locations.data(),
                                           {N, this->input_dim},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .clone();
        torch::Tensor y = torch::from_blob(shuffled_labels.data(),
                                           {N, 1},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .clone();
#else
        torch::Tensor x = torch::from_blob(shuffled_locations.data(),
                                           {N, this->input_dim},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                              .clone();
        torch::Tensor y = torch::from_blob(shuffled_labels.data(),
                                           {N, 1},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                              .clone();
#endif

        // ====== Step 2: 初始化优化器 ======
        float learn_rate = this->learning_rate;
        float min_lr = 1e-6; // 新增最小学习率限制
        torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));

        double best_loss = std::numeric_limits<double>::max();
        int no_improve_count = 0;

        // ====== Step 2.1: 训练 step，加入多样性约束 ======
        auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
        {
            optimizer.zero_grad();

            auto preds = this->forward(xb);
            auto mse = torch::mse_loss(preds, yb);
            auto var_pred = torch::var(preds);

            float lambda_diversity = 0.1; // 多样性权重，可调
            // auto loss = mse - lambda_diversity * var_pred;
            auto loss = mse;

            loss.backward();
            optimizer.step();

            // 监控预测范围
            // std::cout << "Pred range: [" << preds.min().item<float>()
            //           << ", " << preds.max().item<float>() << "]" << std::endl;

            return loss.item<double>();
        };

        // ====== Step 3: 训练循环 ======
        int epoch = 0;
        if (N > 64000000) // 大数据集模式
        {
            size_t batch_num = 4;
            bool training_done = false;

            while (!training_done)
            {
                try
                {
                    auto x_chunks = x.chunk(batch_num, 0);
                    auto y_chunks = y.chunk(batch_num, 0);

                    while (true)
                    {
                        double epoch_loss = 0.0;
                        for (size_t i = 0; i < batch_num; i++)
                        {
                            epoch_loss += train_step(x_chunks[i], y_chunks[i]);
                            // if(epoch > 500) return;
                            epoch++;
                        }
                        epoch_loss /= batch_num;

                        // std::cout << "Loss: " << epoch_loss << " - LR: " << learn_rate << std::endl;

                        double improvement_ratio = (best_loss - epoch_loss) / best_loss;
                        if (improvement_ratio > min_delta_ratio)
                        {
                            best_loss = epoch_loss;
                            no_improve_count = 0;
                        }
                        else
                        {
                            no_improve_count++;
                            if (no_improve_count >= patience)
                            {
                                if (learn_rate > min_lr)
                                {
                                    learn_rate *= 0.5;
                                    optimizer = torch::optim::Adam(this->parameters(),
                                                                   torch::optim::AdamOptions(learn_rate));
                                    // std::cout << "学习率下降到 " << learn_rate << std::endl;
                                    no_improve_count = 0;
                                }
                                else
                                {
                                    // std::cout << "早停触发，结束训练。" << std::endl;
                                    training_done = true;
                                    return;
                                }
                            }
                        }

                        if (epoch_loss <= target_loss)
                        {
                            // std::cout << "达到目标损失值，结束训练。" << std::endl;
                            training_done = true;
                            return;
                        }
                    }
                }
                catch (const c10::Error &e)
                {
                    // std::cerr << "显存不足，batch_num 从 " << batch_num
                    //           << " 增加到 " << batch_num * 2 << std::endl;
                    batch_num *= 2;
                }
            }
        }
        else // 小数据集模式
        {
            while (true)
            {
                double loss_value = train_step(x, y);
                // if(epoch > 500) return;
                epoch++;
                // std::cout << "Loss: " << loss_value << " - LR: " << learn_rate << std::endl;

                double improvement_ratio = (best_loss - loss_value) / best_loss;
                if (improvement_ratio > min_delta_ratio)
                {
                    best_loss = loss_value;
                    no_improve_count = 0;
                }
                else
                {
                    no_improve_count++;
                    if (no_improve_count >= patience)
                    {
                        if (learn_rate > min_lr)
                        {
                            learn_rate *= 0.5;
                            optimizer = torch::optim::Adam(this->parameters(),
                                                           torch::optim::AdamOptions(learn_rate));
                            // std::cout << "学习率下降到 " << learn_rate << std::endl;
                            no_improve_count = 0;
                        }
                        else
                        {
                            // std::cout << "早停触发，结束训练。" << std::endl;
                            return;
                        }
                    }
                }

                if (loss_value <= target_loss)
                {
                    // std::cout << "达到目标损失值，结束训练。" << std::endl;
                    return;
                }
            }
        }
    }

//     void train_model(std::vector<float> locations, std::vector<float> labels)
//     {
//         double target_loss = 1e-1;       // 目标损失
//         int patience = 50;               // 容忍无提升次数
//         double min_delta_ratio = 0.0001; // 相对改善幅度阈值
//         long long N = labels.size();

//         // ====== Step 1: 直接使用原始数据（不打乱） ======
// #ifdef use_gpu
//         torch::Tensor x = torch::from_blob(locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
//         torch::Tensor y = torch::from_blob(labels.data(),
//                                            {N, 1},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
// #else
//         torch::Tensor x = torch::from_blob(locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
//         torch::Tensor y = torch::from_blob(labels.data(),
//                                            {N, 1},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
// #endif

//         // ====== Step 2: 初始化优化器 ======
//         float learn_rate = this->learning_rate;
//         float min_lr = 1e-6; // 新增最小学习率限制
//         torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));

//         double best_loss = std::numeric_limits<double>::max();
//         int no_improve_count = 0;

//         // ====== Step 2.1: 训练 step ======
//         auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
//         {
//             optimizer.zero_grad();

//             auto preds = this->forward(xb);
//             auto mse = torch::mse_loss(preds, yb);

//             auto loss = mse; // 保持原逻辑

//             loss.backward();
//             optimizer.step();

//             return loss.item<double>();
//         };

//         // ====== Step 3: 训练循环 ======
//         int epoch = 0;
//         if (N > 64000000) // 大数据集模式
//         {
//             size_t batch_num = 4;
//             bool training_done = false;

//             while (!training_done)
//             {
//                 try
//                 {
//                     auto x_chunks = x.chunk(batch_num, 0);
//                     auto y_chunks = y.chunk(batch_num, 0);

//                     while (true)
//                     {
//                         double epoch_loss = 0.0;
//                         for (size_t i = 0; i < batch_num; i++)
//                         {
//                             epoch_loss += train_step(x_chunks[i], y_chunks[i]);
//                             epoch++;
//                         }
//                         epoch_loss /= batch_num;

//                         double improvement_ratio = (best_loss - epoch_loss) / best_loss;
//                         if (improvement_ratio > min_delta_ratio)
//                         {
//                             best_loss = epoch_loss;
//                             no_improve_count = 0;
//                         }
//                         else
//                         {
//                             no_improve_count++;
//                             if (no_improve_count >= patience)
//                             {
//                                 if (learn_rate > min_lr)
//                                 {
//                                     learn_rate *= 0.5;
//                                     optimizer = torch::optim::Adam(this->parameters(),
//                                                                    torch::optim::AdamOptions(learn_rate));
//                                     no_improve_count = 0;
//                                 }
//                                 else
//                                 {
//                                     training_done = true;
//                                     return;
//                                 }
//                             }
//                         }

//                         if (epoch_loss <= target_loss)
//                         {
//                             training_done = true;
//                             return;
//                         }
//                     }
//                 }
//                 catch (const c10::Error &e)
//                 {
//                     batch_num *= 2;
//                 }
//             }
//         }
//         else // 小数据集模式
//         {
//             while (true)
//             {
//                 double loss_value = train_step(x, y);
//                 epoch++;

//                 double improvement_ratio = (best_loss - loss_value) / best_loss;
//                 if (improvement_ratio > min_delta_ratio)
//                 {
//                     best_loss = loss_value;
//                     no_improve_count = 0;
//                 }
//                 else
//                 {
//                     no_improve_count++;
//                     if (no_improve_count >= patience)
//                     {
//                         if (learn_rate > min_lr)
//                         {
//                             learn_rate *= 0.5;
//                             optimizer = torch::optim::Adam(this->parameters(),
//                                                            torch::optim::AdamOptions(learn_rate));
//                             no_improve_count = 0;
//                         }
//                         else
//                         {
//                             return;
//                         }
//                     }
//                 }

//                 if (loss_value <= target_loss)
//                 {
//                     return;
//                 }
//             }
//         }
//     }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

#endif