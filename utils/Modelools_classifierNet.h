#ifndef MODELTOOLS_CLASSIFIERNET_H
#define MODELTOOLS_CLASSIFIERNET_H
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
#include "../entities/Point.h"
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

// ===================== 分类网络 =====================
struct NetClassifier : torch::nn::Module
{
private:
    friend class boost::serialization::access;

    template <class Archive>
    void save(Archive &ar, const unsigned int /*version*/) const
    {
        ar & input_dim & width & num_classes & learning_rate & SCALE;
        ar & max_error & min_error;
        ar & train_min_vals & train_max_vals;
        ar & normalize_zorder & z_min & z_max;

        // fc1 参数
        ar & boost::serialization::make_array(w1,   width * input_dim);
        ar & boost::serialization::make_array(b1,   width);
        for (int d = 0; d < input_dim; ++d)
            ar & boost::serialization::make_array(w1_d[d], width);

        // fc2 参数（按类存）
        for (int c = 0; c < num_classes; ++c)
            ar & boost::serialization::make_array(w2[c], width);
        ar & boost::serialization::make_array(b2, num_classes);

        // 缓存（保持结构对齐）
        ar & boost::serialization::make_array(w1_,  width);
        ar & boost::serialization::make_array(b1_,  width);
        ar & boost::serialization::make_array(w1__, width);
        ar & boost::serialization::make_array(w2_buf, width);

        // 加密参数（与原回归版风格一致）
        for (int d = 0; d < 8; ++d) ar & enc_w1_d[d];
        ar & enc_b1;
        ar & enc_w2_by_class;   // [num_classes][width]
        ar & enc_b2_by_class;   // [num_classes]
    }

    template <class Archive>
    void load(Archive &ar, const unsigned int /*version*/)
    {
        ar & input_dim & width & num_classes & learning_rate & SCALE;
        ar & max_error & min_error;
        ar & train_min_vals & train_max_vals;
        ar & normalize_zorder & z_min & z_max;

        cleanup();
        allocate_memory();

        // fc1
        ar & boost::serialization::make_array(w1,   width * input_dim);
        ar & boost::serialization::make_array(b1,   width);
        for (int d = 0; d < input_dim; ++d)
            ar & boost::serialization::make_array(w1_d[d], width);

        // fc2
        for (int c = 0; c < num_classes; ++c)
            ar & boost::serialization::make_array(w2[c], width);
        ar & boost::serialization::make_array(b2, num_classes);

        // 缓存
        ar & boost::serialization::make_array(w1_,  width);
        ar & boost::serialization::make_array(b1_,  width);
        ar & boost::serialization::make_array(w1__, width);
        ar & boost::serialization::make_array(w2_buf, width);

        // 加密参数
        for (int d = 0; d < 8; ++d) ar & enc_w1_d[d];
        ar & enc_b1;
        ar & enc_w2_by_class;
        ar & enc_b2_by_class;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // ============= 内存管理 =============
    void cleanup()
    {
        if (w1)    _mm_free(w1);
        if (b1)    _mm_free(b1);
        if (w1_)   _mm_free(w1_);
        if (b1_)   _mm_free(b1_);
        if (w1__)  _mm_free(w1__);
        if (w2_buf)_mm_free(w2_buf);

        if (w1_d) {
            for (int d = 0; d < input_dim; ++d)
                if (w1_d[d]) _mm_free(w1_d[d]);
            delete[] w1_d;
            w1_d = nullptr;
        }

        if (w2) {
            for (int c = 0; c < num_classes; ++c)
                if (w2[c]) _mm_free(w2[c]);
            delete[] w2;
            w2 = nullptr;
        }

        if (b2) _mm_free(b2);

        w1 = b1 = w1_ = b1_ = w1__ = w2_buf = nullptr;
        b2 = nullptr;
    }

    void allocate_memory()
    {
        // fc1
        w1 = (float*)_mm_malloc(width * input_dim * sizeof(float), 32);
        b1 = (float*)_mm_malloc(width * sizeof(float), 32);

        w1_d = new float*[input_dim];
        for (int d = 0; d < input_dim; ++d)
            w1_d[d] = (float*)_mm_malloc(width * sizeof(float), 32);

        // 缓存
        w1_    = (float*)_mm_malloc(width * sizeof(float), 32);
        b1_    = (float*)_mm_malloc(width * sizeof(float), 32);
        w1__   = (float*)_mm_malloc(width * sizeof(float), 32);
        w2_buf = (float*)_mm_malloc(width * sizeof(float), 32);

        // fc2
        w2 = new float*[num_classes];
        for (int c = 0; c < num_classes; ++c)
            w2[c] = (float*)_mm_malloc(width * sizeof(float), 32);
        b2 = (float*)_mm_malloc(num_classes * sizeof(float), 32);
    }

    // 加密选择器：根据明文 bit 选择密文 (b? ct_true : ct_false)
    // 使用 enc_b 与 enc_(1-b) + SM 实现
    static Ciphertext ct_select_bit(Integer bit, const Ciphertext& ct_true, const Ciphertext& ct_false, PaillierFast& p)
    {
        Integer ibit(bit);
        Integer ione_minus(1 - bit);
        Ciphertext enc_b     = p.encrypt(ibit);
        Ciphertext enc_1_b   = p.encrypt(ione_minus);
        Ciphertext t = SMrun(ct_true,  enc_b,   p);
        Ciphertext f = SMrun(ct_false, enc_1_b, p);
        return t + f;
    }

    // 安全 ReLU：res = max(0, x)
    static void secure_relu(Ciphertext& out, const Ciphertext& in, PaillierFast& p)
    {
        Integer izero(0);
        Ciphertext enc0 = p.encrypt(izero);
        // SIC(enc0, in) = 1  当 0 <= in
        Integer flag = SICrun(enc0, in, p); // 明文 0/1
        out = ct_select_bit(flag, in, enc0, p);
    }

public:
    // ======= 公有配置 =======
    int   input_dim = 2;               // 2 ~ 6
    int   width     = 0;               // 隐层宽度
    int   num_classes = 2;             // 1 ~ 10
    float learning_rate = Constants::LEARNING_RATE;
    long  SCALE = 1000000L;            // 放大系数（与 Point::encrypt_point 一致）

    int max_error = 0, min_error = 0;
    bool normalize_zorder = false;
    uint64_t z_min = 0, z_max = 0;
    std::vector<float> train_min_vals, train_max_vals;

    // Torch 层
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    // fc1 原生参数（SIMD 友好）
    float *w1 = nullptr;     // [width * input_dim]  行主（i * input_dim + d）
    float *b1 = nullptr;     // [width]
    float **w1_d = nullptr;  // [input_dim][width]

    // fc2 原生参数
    float **w2 = nullptr;    // [num_classes][width]
    float *b2  = nullptr;    // [num_classes]

    // 缓存区（为与你原结构保持一致）
    float *w1_ = nullptr, *b1_ = nullptr, *w1__ = nullptr, *w2_buf = nullptr;

    // 加密权重
    std::vector<Ciphertext> enc_w1_d[8];                // 每维一组（上限 8）
    std::vector<Ciphertext> enc_b1;                     // [width]
    std::vector<std::vector<Ciphertext>> enc_w2_by_class; // [num_classes][width]
    std::vector<Ciphertext> enc_b2_by_class;            // [num_classes]

    std::chrono::milliseconds sic_time{0};
    std::chrono::milliseconds sm_time{0};

    // ======= 构造/析构 =======
    NetClassifier() {}

    // NetClassifier(const NetClassifier &other)
    // {
    //     // 1. 基本成员变量
    //     input_dim = other.input_dim;
    //     width = other.width;
    //     num_classes = other.num_classes;
    //     learning_rate = other.learning_rate;
    //     SCALE = other.SCALE;
    //     max_error = other.max_error;
    //     min_error = other.min_error;
    //     normalize_zorder = other.normalize_zorder;
    //     z_min = other.z_min;
    //     z_max = other.z_max;
    //     train_min_vals = other.train_min_vals;
    //     train_max_vals = other.train_max_vals;

    //     // ⚠️ 不再拷贝 fc1/fc2，保持默认 nullptr 即可

    //     // 2. 分配并拷贝 w1 / b1
    //     w1 = (other.w1) ? new float[width * input_dim] : nullptr;
    //     if (w1)
    //         std::copy(other.w1, other.w1 + width * input_dim, w1);

    //     b1 = (other.b1) ? new float[width] : nullptr;
    //     if (b1)
    //         std::copy(other.b1, other.b1 + width, b1);

    //     // 3. 拷贝 w1_d (二维)
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

    //     // 4. 拷贝 w2 / b2
    //     if (other.w2)
    //     {
    //         w2 = new float *[num_classes];
    //         for (int c = 0; c < num_classes; c++)
    //         {
    //             w2[c] = new float[width];
    //             std::copy(other.w2[c], other.w2[c] + width, w2[c]);
    //         }
    //     }
    //     else
    //     {
    //         w2 = nullptr;
    //     }

    //     b2 = (other.b2) ? new float[num_classes] : nullptr;
    //     if (b2)
    //         std::copy(other.b2, other.b2 + num_classes, b2);

    //     // 5. 缓存区
    //     w1_ = (other.w1_) ? new float[width * input_dim] : nullptr;
    //     if (w1_)
    //         std::copy(other.w1_, other.w1_ + width * input_dim, w1_);

    //     b1_ = (other.b1_) ? new float[width] : nullptr;
    //     if (b1_)
    //         std::copy(other.b1_, other.b1_ + width, b1_);

    //     w1__ = (other.w1__) ? new float[width * input_dim] : nullptr;
    //     if (w1__)
    //         std::copy(other.w1__, other.w1__ + width * input_dim, w1__);

    //     w2_buf = (other.w2_buf) ? new float[num_classes * width] : nullptr;
    //     if (w2_buf)
    //         std::copy(other.w2_buf, other.w2_buf + num_classes * width, w2_buf);

    //     // 6. 加密权重
    //     for (int d = 0; d < 8; d++)
    //     {
    //         enc_w1_d[d] = other.enc_w1_d[d];
    //     }
    //     enc_b1 = other.enc_b1;
    //     enc_w2_by_class = other.enc_w2_by_class;
    //     enc_b2_by_class = other.enc_b2_by_class;

    //     // 7. 时间统计
    //     sic_time = other.sic_time;
    //     sm_time = other.sm_time;
    // }

    NetClassifier(const NetClassifier &other)
    {
        // 基本字段
        input_dim = other.input_dim;
        width = other.width;
        num_classes = other.num_classes;
        learning_rate = other.learning_rate;
        SCALE = other.SCALE;
        max_error = other.max_error;
        min_error = other.min_error;
        normalize_zorder = other.normalize_zorder;
        z_min = other.z_min;
        z_max = other.z_max;
        train_min_vals = other.train_min_vals;
        train_max_vals = other.train_max_vals;

        // ⚠️ 显式置空 fc1/fc2
        fc1 = nullptr;
        fc2 = nullptr;

        // w1 / b1
        w1 = (other.w1) ? new float[width * input_dim] : nullptr;
        if (w1)
            std::copy(other.w1, other.w1 + width * input_dim, w1);

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

        // w2
        if (other.w2)
        {
            w2 = new float *[num_classes];
            for (int c = 0; c < num_classes; c++)
            {
                w2[c] = new float[width];
                std::copy(other.w2[c], other.w2[c] + width, w2[c]);
            }
        }
        else
        {
            w2 = nullptr;
        }

        b2 = (other.b2) ? new float[num_classes] : nullptr;
        if (b2)
            std::copy(other.b2, other.b2 + num_classes, b2);

        // 缓存
        w1_ = (other.w1_) ? new float[width * input_dim] : nullptr;
        if (w1_)
            std::copy(other.w1_, other.w1_ + width * input_dim, w1_);

        b1_ = (other.b1_) ? new float[width] : nullptr;
        if (b1_)
            std::copy(other.b1_, other.b1_ + width, b1_);

        w1__ = (other.w1__) ? new float[width * input_dim] : nullptr;
        if (w1__)
            std::copy(other.w1__, other.w1__ + width * input_dim, w1__);

        w2_buf = (other.w2_buf) ? new float[num_classes * width] : nullptr;
        if (w2_buf)
            std::copy(other.w2_buf, other.w2_buf + num_classes * width, w2_buf);

        // 加密权重
        for (int d = 0; d < 8; d++)
        {
            enc_w1_d[d] = other.enc_w1_d[d];
        }
        enc_b1 = other.enc_b1;
        enc_w2_by_class = other.enc_w2_by_class;
        enc_b2_by_class = other.enc_b2_by_class;

        // 时间
        sic_time = other.sic_time;
        sm_time = other.sm_time;
    }

    NetClassifier(int input_dim_, int width_, int num_classes_)
        : input_dim(input_dim_),
          width(std::min(width_, Constants::HIDDEN_LAYER_WIDTH)),
          num_classes(num_classes_)
    {
        // torch 模块
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, width));
        fc2 = register_module("fc2", torch::nn::Linear(width, num_classes));
        torch::nn::init::kaiming_uniform_(fc1->weight);
        torch::nn::init::kaiming_uniform_(fc2->weight);

        allocate_memory();
    }

    ~NetClassifier() { cleanup(); }

    // ======= Torch forward：返回 logits（不做 softmax）=======
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);  // [N, num_classes] logits
        // x = torch::softmax(x, 1);  // ✅ 添加Softmax激活
        return x;
    }

    // ======= 从 Torch 参数同步到原生数组（训练后调用）=======
    void get_parameters()
    {
        // 顺序：w1, b1, w2, b2
        torch::Tensor p1 = this->parameters()[0]; // fc1.weight [width, input_dim]
        torch::Tensor p2 = this->parameters()[1]; // fc1.bias   [width]
        torch::Tensor p3 = this->parameters()[2]; // fc2.weight [num_classes, width]
        torch::Tensor p4 = this->parameters()[3]; // fc2.bias   [num_classes]

        p1 = p1.reshape({width, input_dim});
        for (int i = 0; i < width; ++i) {
            for (int d = 0; d < input_dim; ++d) {
                float v = p1[i][d].item().toFloat();
                w1[i * input_dim + d] = (v * SCALE) / SCALE;
                w1_d[d][i]           = (v * SCALE) / SCALE;
                cout<<w1_d[d][i]<<" ";
            }
        }
        cout<<endl;

        for (int i = 0; i < width; ++i) {
            float v = p2[i].item().toFloat();
            b1[i]  = (v * SCALE) / SCALE;
            b1_[i] = b1[i];
            cout<<b1[i]<<" ";
        }
        cout<<endl;
        for (int c = 0; c < num_classes; ++c)
        {    for (int i = 0; i < width; ++i)
            {
                w2[c][i] = (p3[c][i].item().toFloat() * SCALE) / SCALE;
                cout<<w2[c][i]<<" ";
            }
        }
        cout<<endl;
        for (int c = 0; c < num_classes; ++c)
        {
            b2[c] = (p4[c].item().toFloat() * SCALE) / SCALE;
            cout<<b2[c]<<" ";
        }
        cout<<endl;
    }
    // ======= 明文推理（单点）：返回 argmax 类别 =======
    // int predict(const Point& point) const
    // {
    //     const auto& x = point.coords;
    //     if ((int)x.size() != input_dim)
    //         throw std::invalid_argument("Point dimension does not match network input dimension");

    //     // 隐层
    //     std::vector<double> h(width);
    //     for (int i = 0; i < width; ++i) {
    //         double s = 0.0;
    //         // 访问模式：遍历维度 *w1_d[d][i]（SIMD 友好）
    //         for (int d = 0; d < input_dim; ++d)
    //             s += (double)x[d] * (double)w1_d[d][i];
    //         s += (double)b1[i];
    //         h[i] = (s > 0.0) ? s : 0.0;
    //     }

    //     // 输出 logits & argmax
    //     int best = 0;
    //     double bestv = -std::numeric_limits<double>::infinity();
    //     for (int c = 0; c < num_classes; ++c) {
    //         double s = 0.0;
    //         for (int i = 0; i < width; ++i)
    //             s += h[i] * (double)w2[c][i];
    //         s += (double)b2[c];
    //         if (s > bestv) { bestv = s; best = c; }
    //     }
    //     return best;
    // }
    int predict(const Point &point) const
    {
        const auto &x = point.coords;
        if ((int)x.size() != input_dim)
            throw std::invalid_argument("Point dimension does not match network input dimension");

        long SCALE(1000000); // 定点缩放因子，例如 1e6

        // ---------- 1) 输入转整数 ----------
        std::vector<Integer> X(input_dim);
        for (int d = 0; d < input_dim; ++d)
        {
            X[d] = Integer(static_cast<long long>((x[d] * SCALE))); // 单位 S
        }

        // ---------- 2) 隐层 ----------
        std::vector<Integer> H(width, 0);
        for (int i = 0; i < width; ++i)
        {
            Integer s = 0;
            for (int d = 0; d < input_dim; ++d)
            {
                Integer W = Integer(static_cast<long long>((w1_d[d][i] * SCALE))); // 单位 S
                s += X[d] * W;                                                          // S*S = S^2
            }
            Integer B = Integer(static_cast<long long>((b1[i] * SCALE * SCALE))); // 偏置单位 S^2
            s += B;

            // ReLU
            H[i] = (s > 0 ? s : 0); // 单位 S^2
        }

        // ---------- 3) 输出层 ----------
        std::vector<Integer> logits(num_classes, 0);
        for (int c = 0; c < num_classes; ++c)
        {
            Integer s = 0;
            for (int i = 0; i < width; ++i)
            {
                Integer W = Integer(static_cast<long long>((w2[c][i] * SCALE))); // 单位 S
                s += H[i] * W;                                                        // S^2 * S = S^3
            }
            Integer B = Integer(static_cast<long long>((b2[c] * SCALE * SCALE * SCALE))); // 偏置 S^3
            s += B;
            logits[c] = s;
        }

        // ---------- 4) argmax ----------
        int best = 0;
        Integer bestv = logits[0];
        for (int c = 1; c < num_classes; ++c)
        {
            if (logits[c] > bestv)
            {
                bestv = logits[c];
                best = c;
            }
        }

        return best;
    }

    // ======= 初始化加密参数（训练后调用）=======
    void init_encrypted_params(PaillierFast& p)
    {
        auto enc_param = [&](float val, int scale) -> Ciphertext {
            // 与你回归版一致：w(×SCALE^1), b1(×SCALE^2), w2(×SCALE^1), b2(×SCALE^3)
            // 我们按 scale 次乘以 SCALE
            mpf_class SCALED(SCALE), VAL(val);
            for (int i = 0; i < scale; ++i) VAL *= SCALED;
            Integer scaled = mpz_class(VAL);
            return p.encrypt(scaled);
        };

        // enc_w1_d / enc_b1
        for (int d = 0; d < input_dim; ++d)
            enc_w1_d[d] = std::vector<Ciphertext>(width);

        enc_b1 = std::vector<Ciphertext>(width);

        for (int d = 0; d < input_dim; ++d)
            for (int i = 0; i < width; ++i)
                enc_w1_d[d][i] = enc_param(w1_d[d][i], 1);
        for (int i = 0; i < width; ++i)
            enc_b1[i] = enc_param(b1[i], 2);

        // enc_w2_by_class / enc_b2_by_class
        enc_w2_by_class.assign(num_classes, std::vector<Ciphertext>(width));
        enc_b2_by_class.assign(num_classes, Ciphertext());

        for (int c = 0; c < num_classes; ++c) {
            for (int i = 0; i < width; ++i)
                enc_w2_by_class[c][i] = enc_param(w2[c][i], 1);
            enc_b2_by_class[c] = enc_param(b2[c], 3);
        }
    }

    // ======= 密文推理（单点）：返回 argmax 类别（全程同态比较）=======
    // 返回：密文类别编号（0..num_classes-1 的密文）
    Ciphertext enc_predict(const Point &point, PaillierFast &p) const
    {
        const auto &X = point.enc_coords;
        if ((int)X.size() != input_dim)
            throw std::invalid_argument("Encrypted point dimension does not match network input dimension");

        Integer izero(0);
        Ciphertext enc0 = p.encrypt(izero);

        // 1) 隐层：线性 + ReLU（密文）
        std::vector<Ciphertext> H(width, enc0);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < width; ++i)
        {
            // int tid = omp_get_thread_num();
            // #pragma omp critical
            // {
            //     std::cout << "Iteration " << i << " executed by thread " << tid << std::endl;
            // }
            Ciphertext s = enc0;
            for (int d = 0; d < input_dim; ++d)
                s = s + SMrun(X[d], enc_w1_d[d][i], p);
            s = s + enc_b1[i];
            secure_relu(H[i], s, p);
        }

        // 2) 各类 logits（密文）
        std::vector<Ciphertext> logits(num_classes, enc0);

        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < num_classes; ++c)
        {
            // int tid = omp_get_thread_num();
            // #pragma omp critical
            // {
            //     std::cout << "Iteration " << c << " executed by thread " << tid << std::endl;
            // }
            Ciphertext s = enc0;
            for (int i = 0; i < width; ++i)
                s = s + SMrun(H[i], enc_w2_by_class[c][i], p);
            logits[c] = s + enc_b2_by_class[c];
        }

        // 3) 同态 argmax：维护“最佳分数密文 + 最佳编号密文”
        Ciphertext best_score = logits[0];
        Ciphertext best_idx = p.encrypt(Integer(0)); // 0-based

        for (int c = 1; c < num_classes; ++c)
        {
            Integer flag = SICrun(best_score, logits[c], p); // 1 if best_score <= logits[c]
            // best_score = flag ? logits[c] : best_score
            if(flag == 1) 
            {
                best_score = logits[c];
                best_idx = p.encrypt(c);
            }
        }

        return best_idx; // 返回密文类别编号（0..C-1）
    }

    // ======= 保存/加载 =======
    void save_model(const std::string& filepath) const
    {
        boost::filesystem::path path(filepath);
        boost::filesystem::create_directories(path.parent_path());

        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open())
            throw std::runtime_error("Failed to open file for saving model: " + filepath);
        boost::archive::binary_oarchive oa(ofs);
        oa << *this;
        ofs.close();
        std::cout << "Classifier model saved to: " << filepath << std::endl;
    }

    bool load_model(const std::string& filepath)
    {
        if (!boost::filesystem::exists(filepath)) return false;
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) return false;
        boost::archive::binary_iarchive ia(ifs);
        ia >> *this;
        ifs.close();
        std::cout << "Classifier model loaded from: " << filepath << std::endl;
        return true;
    }

//     void train_model_classification(std::vector<float> locations, std::vector<__uint128_t> labels)
//     {
//         double target_loss = 1e-1;       // 目标损失
//         int patience = 50;               // 容忍无提升次数
//         double min_delta_ratio = 0.0001; // 相对改善幅度阈值
//         long long N = labels.size();

//         // ====== Step 1: 打乱数据 ======
//         std::vector<long long> indices(N);
//         std::iota(indices.begin(), indices.end(), 0);
//         std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

//         std::vector<float> shuffled_locations;
//         std::vector<__uint128_t> shuffled_labels;
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
//                                            {N},
//                                            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
//                               .clone();
// #else
//         torch::Tensor x = torch::from_blob(shuffled_locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
//         torch::Tensor y = torch::from_blob(shuffled_labels.data(),
//                                            {N},
//                                            torch::TensorOptions().dtype(torch::kInt64))
//                               .clone();
// #endif

//         // ====== Step 2: 初始化优化器和损失 ======
//         float learn_rate = this->learning_rate;
//         float min_lr = 1e-6;
//         torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));
//         torch::nn::CrossEntropyLoss criterion;

//         double best_loss = std::numeric_limits<double>::max();
//         int no_improve_count = 0;

//         // ====== Step 2.1: 训练 step ======
//         auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
//         {
//             optimizer.zero_grad();

//             auto logits = this->forward(xb);   // [batch, num_classes]
//             auto loss = criterion(logits, yb); // yb: [batch] int64

//             loss.backward();
//             optimizer.step();

//             // 监控预测分布范围
//             auto probs = torch::softmax(logits, 1);
//             std::cout << "Pred prob range: [" << probs.min().item<float>()
//                       << ", " << probs.max().item<float>() << "]" << std::endl;

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
//                             epoch_loss += train_step(x_chunks[i], y_chunks[i]);

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
//                                 if (learn_rate > min_lr)
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
//                         if (learn_rate > min_lr)
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

    void train_model_classification(std::vector<float> locations, std::vector<__uint128_t> labels)
    {
        double target_loss = 1e-1;       // 目标损失
        int patience = 50;               // 容忍无提升次数
        double min_delta_ratio = 0.0001; // 相对改善幅度阈值
        long long N = static_cast<long long>(labels.size());

        // ====== Step 0: 设备设置 ======
#ifdef use_gpu
        this->to(torch::kCUDA); // ✅ 确保模型在 CUDA
#endif
        this->train(); // 进入训练模式

        // ====== Step 1: 打乱数据 ======
        std::vector<long long> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 gen(42);
        // std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<float> shuffled_locations;
        std::vector<__uint128_t> shuffled_labels;
        shuffled_locations.reserve(locations.size());
        shuffled_labels.reserve(labels.size());

        for (auto idx : indices)
        {
            shuffled_locations.insert(shuffled_locations.end(),
                                      locations.begin() + idx * this->input_dim,
                                      locations.begin() + (idx + 1) * this->input_dim);
            shuffled_labels.push_back(labels[idx]);
        }

        // 将 __uint128_t 标签安全转换为 int64_t
        std::vector<int64_t> labels_i64(N);
        for (long long i = 0; i < N; ++i)
        {
            __uint128_t v = shuffled_labels[i];
            // 基本校验：必须落在 [0, num_classes-1]
            if (v >= static_cast<__uint128_t>(num_classes))
                throw std::invalid_argument("Label out of range for num_classes");
            labels_i64[i] = static_cast<int64_t>(static_cast<uint64_t>(v));
        }

#ifdef use_gpu
        torch::Tensor x = torch::from_blob(shuffled_locations.data(),
                                           {N, this->input_dim},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .clone();
        torch::Tensor y = torch::from_blob(labels_i64.data(),
                                           {N},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
                              .clone();
#else
        torch::Tensor x = torch::from_blob(shuffled_locations.data(),
                                           {N, this->input_dim},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                              .clone();
        torch::Tensor y = torch::from_blob(labels_i64.data(),
                                           {N},
                                           torch::TensorOptions().dtype(torch::kInt64))
                              .clone();
#endif

        // ====== Step 2: 初始化优化器和损失 ======
        float learn_rate = this->learning_rate;
        float min_lr = 1e-6f;
        torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));

        // 计算类别权重：N / (C * n_c)
        std::vector<long long> class_counts(num_classes, 0);
        for (auto v : labels_i64)
            class_counts[static_cast<int>(v)]++;

        std::vector<float> class_weights(num_classes, 0.0f);
        for (int c = 0; c < num_classes; ++c)
            class_weights[c] = (class_counts[c] > 0)
                                   ? static_cast<float>(N) / (num_classes * static_cast<float>(class_counts[c]))
                                   : 0.0f;

        std::cout << "Class distribution and weights:\n";
        for (int c = 0; c < num_classes; ++c)
            std::cout << "  Class " << c << ": " << class_counts[c]
                      << " samples, weight = " << class_weights[c] << "\n";

#ifdef use_gpu
        torch::Tensor weight_tensor = torch::from_blob(class_weights.data(),
                                                       {num_classes},
                                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                                          .clone();
#else
        torch::Tensor weight_tensor = torch::from_blob(class_weights.data(),
                                                       {num_classes},
                                                       torch::TensorOptions().dtype(torch::kFloat32))
                                          .clone();
#endif
        // 交叉熵输入必须是 logits（我们已在 forward 修正）
        torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().weight(weight_tensor));

        double best_loss = std::numeric_limits<double>::max();
        int no_improve_count = 0;

        // ====== Step 2.1: 训练 step ======
        auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
        {
            optimizer.zero_grad();

            auto logits = this->forward(xb);   // [batch, num_classes] 纯 logits
            auto loss = criterion(logits, yb); // yb: [batch] int64

            loss.backward();
            optimizer.step();

            // 监控（这里再 softmax）
            auto probs = torch::softmax(logits, 1);
            // std::cout << "Pred prob range: [" << probs.min().item<float>()
            //           << ", " << probs.max().item<float>() << "]\n";

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
                        epoch_loss /= static_cast<double>(batch_num);
                        // std::cout << "Loss: " << epoch_loss << " - LR: " << learn_rate << std::endl;

                        double improvement_ratio = (best_loss - epoch_loss) / std::max(best_loss, 1e-12);
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
                                    learn_rate *= 0.5f;
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
                catch (const c10::Error &)
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

                double improvement_ratio = (best_loss - loss_value) / std::max(best_loss, 1e-12);
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
                            learn_rate *= 0.5f;
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
//     void train_model_classification(std::vector<float> locations, std::vector<__uint128_t> labels)
//     {
//         double target_loss = 1e-1;       // 目标损失
//         int patience = 50;               // 容忍无提升次数
//         double min_delta_ratio = 0.0001; // 相对改善幅度阈值
//         long long N = static_cast<long long>(labels.size());

//         // ====== Step 0: 设备设置 ======
// #ifdef use_gpu
//         this->to(torch::kCUDA); // ✅ 确保模型在 CUDA
// #endif
//         this->train(); // 进入训练模式

//         // ====== Step 1: 直接使用传入数据（不打乱） ======

//         // 将 __uint128_t 标签安全转换为 int64_t
//         std::vector<int64_t> labels_i64(N);
//         for (long long i = 0; i < N; ++i)
//         {
//             __uint128_t v = labels[i];
//             if (v >= static_cast<__uint128_t>(num_classes))
//                 throw std::invalid_argument("Label out of range for num_classes");
//             labels_i64[i] = static_cast<int64_t>(static_cast<uint64_t>(v));
//         }

// #ifdef use_gpu
//         torch::Tensor x = torch::from_blob(locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                               .clone();
//         torch::Tensor y = torch::from_blob(labels_i64.data(),
//                                            {N},
//                                            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
//                               .clone();
// #else
//         torch::Tensor x = torch::from_blob(locations.data(),
//                                            {N, this->input_dim},
//                                            torch::TensorOptions().dtype(torch::kFloat32))
//                               .clone();
//         torch::Tensor y = torch::from_blob(labels_i64.data(),
//                                            {N},
//                                            torch::TensorOptions().dtype(torch::kInt64))
//                               .clone();
// #endif

//         // ====== Step 2: 初始化优化器和损失 ======
//         float learn_rate = this->learning_rate;
//         float min_lr = 1e-6f;
//         torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learn_rate));

//         // 计算类别权重：N / (C * n_c)
//         std::vector<long long> class_counts(num_classes, 0);
//         for (auto v : labels_i64)
//             class_counts[static_cast<int>(v)]++;

//         std::vector<float> class_weights(num_classes, 0.0f);
//         for (int c = 0; c < num_classes; ++c)
//             class_weights[c] = (class_counts[c] > 0)
//                                    ? static_cast<float>(N) / (num_classes * static_cast<float>(class_counts[c]))
//                                    : 0.0f;

//         std::cout << "Class distribution and weights:\n";
//         for (int c = 0; c < num_classes; ++c)
//             std::cout << "  Class " << c << ": " << class_counts[c]
//                       << " samples, weight = " << class_weights[c] << "\n";

// #ifdef use_gpu
//         torch::Tensor weight_tensor = torch::from_blob(class_weights.data(),
//                                                        {num_classes},
//                                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
//                                           .clone();
// #else
//         torch::Tensor weight_tensor = torch::from_blob(class_weights.data(),
//                                                        {num_classes},
//                                                        torch::TensorOptions().dtype(torch::kFloat32))
//                                           .clone();
// #endif

//         torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().weight(weight_tensor));

//         double best_loss = std::numeric_limits<double>::max();
//         int no_improve_count = 0;

//         // ====== Step 2.1: 训练 step ======
//         auto train_step = [&](const torch::Tensor &xb, const torch::Tensor &yb)
//         {
//             optimizer.zero_grad();

//             auto logits = this->forward(xb);   // [batch, num_classes]
//             auto loss = criterion(logits, yb); // yb: [batch]

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
//                         epoch_loss /= static_cast<double>(batch_num);

//                         double improvement_ratio = (best_loss - epoch_loss) / std::max(best_loss, 1e-12);
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
//                                     learn_rate *= 0.5f;
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
//                 catch (const c10::Error &)
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

//                 double improvement_ratio = (best_loss - loss_value) / std::max(best_loss, 1e-12);
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
//                             learn_rate *= 0.5f;
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
};

#endif // CLASSIFIER_NET_H
