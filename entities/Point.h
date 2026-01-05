// #ifndef POINT_H
// #define POINT_H
// #include <vector>
// #include <string.h>
// #include <string>
// #include <gmp.h>
// #include <gmpxx.h>
// #include <unordered_set>
// #include "../Paillier/paillier.h"
// #include <ophelib/paillier_fast.h>
// #include <boost/serialization/access.hpp>
// #include <boost/serialization/vector.hpp>
// #include <boost/serialization/shared_ptr.hpp>
// using namespace ophelib;
// using namespace std;
// class Point
// {

// public:

//     float index;
//     int dim; // 新增维度字段
//     std::vector<float> coords; // 替换原来的x,y
//     std::vector<Ciphertext> enc_coords; // 替换原来的enc_x, enc_y
//     long long x_i;
//     long long y_i;
//     long long curve_val;

//     float normalized_curve_val;

//     float temp_dist = 0.0;

//     Ciphertext enc_temp_dist;


//     //int res = 0;

//     // 构造函数需要修改
//     Point(const std::vector<float>& coords);
//     Point(const std::vector<Ciphertext>& enc_coords);
//     Point(const std::vector<float>& coords, const std::vector<Ciphertext>& enc_coords);
//     Point();

//     void encrypt_point(PaillierFast&);//加密点函数

//     bool operator == (const Point& point) const;
//     // float cal_dist(Point);
//     float cal_dist(const Point& point) ;

//     Ciphertext enc_cal_dist(const Point& point, PaillierFast& paillier) ;

//     Ciphertext enc_cal_dist(Point, PaillierFast&);

//     void print();
//     void init();
//     static vector<Point> get_points(vector<Point>, int);
//     static vector<Point> get_inserted_points(int, long long);

//     string get_self();

// private:
//     friend class boost::serialization::access;
//     template <class Archive>
//     void serialize(Archive &ar, const unsigned int version)
//     {
//         // ar & index & x & y;
//         // ar & enc_x & enc_y;
//         ar & index;
//         ar & dim & coords & enc_coords;
//         ar & x_i & y_i;
//         ar & curve_val & normalized_curve_val;
//         ar & temp_dist & enc_temp_dist;
//     }
// };
// // 自定义哈希函数
// struct PointHash
// {
//     std::size_t operator()(const Point &p) const
//     {
//         // // 使用 std::hash 对 x 和 y 进行哈希，然后组合
//         // auto hashX = std::hash<float>()(p.x);
//         // auto hashY = std::hash<float>()(p.y);
//         // return hashX ^ (hashY << 1);
//         std::size_t seed;
//         for(int i = 0; i < p.coords.size(); i++)
//         {
//             auto hashP = std::hash<float>()(p.coords[i]);
//         }
//     }
// };

// #endif


#ifndef POINT_H
#define POINT_H
#include <vector>
#include <string>
#include <ophelib/paillier_fast.h>
#include "../Serialize/serialization_helpers.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
using namespace ophelib;
using namespace std;

class Point
{
public:
    int dim;  // 新增：维度字段
    vector<float> coords;  // 替换x,y为向量
    vector<Ciphertext> enc_coords;  // 替换enc_x,enc_y为向量
    
    // 保留其他字段
    float index;
    long long x_i;
    long long y_i;
    long long curve_val;
    float normalized_curve_val;
    float temp_dist = 0.0;
    Ciphertext enc_temp_dist;

    // 构造函数修改为支持多维
    Point(int dim); 
    Point(int dim, PaillierFast &paillier); 
    Point(int dim, const vector<float>& coords);  // 新构造函数
    Point(int dim, const vector<Ciphertext>& enc_coords);  // 新构造函数
    Point(int dim, const vector<float>& coords, const vector<Ciphertext>& enc_coords);  // 新构造函数
    Point(const vector<float>& coords);  // 新构造函数
    Point(const vector<Ciphertext>& enc_coords);  // 新构造函数
    Point(const vector<float>& coords, const vector<Ciphertext>& enc_coords);  // 新构造函数
    Point(float x, float y);  // 保留但标记为过时
    Point();  // 默认构造函数

    // 新增：获取维度
    int get_dim() const { return dim; }
    
    void encrypt_point(PaillierFast& paillier);
    bool operator == (const Point& point) const;
    float cal_dist(const Point& other) const;
    Ciphertext enc_cal_dist(const Point& other, PaillierFast& paillier) const;
    void print();
    void init();
    static vector<Point> get_points(vector<Point> dataset, int num);
    static vector<Point> get_inserted_points(long long num, int dim);
    string get_self();

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & dim & coords & enc_coords;
        ar & index;
        ar & x_i & y_i;
        ar & curve_val & normalized_curve_val;
        ar & temp_dist & enc_temp_dist;
    }
};

struct PointHash {
    std::size_t operator()(const Point &p) const {
        std::size_t seed = 0;
        for (float coord : p.coords) {
            seed ^= std::hash<float>()(coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

#endif