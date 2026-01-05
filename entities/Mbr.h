#ifndef MBR_H
#define MBR_H
#include <vector>
#include <limits>
#include <ophelib/paillier_fast.h>
#include "Point.h"
#include "../agreements/SIC.h"
#include "../agreements/SM.h"
#include "../agreements/SSED.h"
#include "../Serialize/serialization_helpers.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
using namespace ophelib;
using namespace std;

class Mbr
{
public:
    int dim;  // 新增：维度字段
    vector<float> lows;   // 各维度下界
    vector<float> highs;  // 各维度上界
    vector<Ciphertext> enc_lows;   // 加密下界
    vector<Ciphertext> enc_highs;  // 加密上界

    Mbr();  // 默认构造函数
    Mbr(int dim);  // 新构造函数：指定维度
    Mbr(int dim, const vector<float>& lows, const vector<float>& highs);  // 新构造函数

    void init();
    void clear();
    void encrypt_mbr(PaillierFast& paillier);
    void update(const Point& point);
    void update(const Point& point, PaillierFast& paillier);
    void update(const Mbr& other);
    void enc_update(const vector<Ciphertext>& enc_coords, PaillierFast& paillier);
    bool contains(const Point& point) const;
    bool enc_contains(const Point& point, PaillierFast& paillier) const;
    bool strict_contains(const Point& point) const;
    bool enc_strict_contains(const Point& point, PaillierFast& paillier) const;
    bool interact(const Mbr& other) const;
    bool enc_interact(const Mbr& other, PaillierFast& paillier) const;
    // static vector<Mbr> get_mbrs(vector<Point> dataset, float area, int num, float ratio);
    // static vector<Mbr> get_mbrs(vector<Point> dataset, float area, int num, vector<float> ratios);
    static vector<Mbr> get_mbrs(const vector<Point>& dataset, float selectivity, int num);
    float cal_dist(const Point& point) const;
    Ciphertext enc_cal_dist(const Point& point, PaillierFast& paillier) const;
    void print();
    vector<Point> get_corner_points() const;
    vector<Point> enc_get_corner_points() const;
    vector<Point> init_get_corner_points() ;
    static Mbr get_mbr(const Point& point, float knnquerySide, int dim);
    void clean();
    string get_self();

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & dim & lows & highs;
        ar & enc_lows & enc_highs;
    }
};

#endif