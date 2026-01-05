#include "Point.h"
#include <iostream>
#include <cmath>
#include "../agreements/SIC.h"
#include "../agreements/SM.h"
#include "../agreements/SSED.h"
#include "../Serialize/serialization_helpers.hpp"
Point::Point(int dim)
{
    this->dim = dim;
    coords = vector<float>(dim);
    enc_coords = vector<Ciphertext>(dim);
}

Point::Point(int dim, PaillierFast &paillier)
{
    this->dim = dim;
    coords = vector<float>(dim);
    enc_coords = vector<Ciphertext>(dim);
    enc_temp_dist = paillier.encrypt(11000000000000);
}

Point::Point() : dim(0) {}  // 默认维度0

// 新构造函数：指定维度和坐标
Point::Point(int dim, const vector<float>& coords) 
    : dim(dim), coords(coords) {
    enc_coords.resize(dim);
}

// 新构造函数：指定维度和加密坐标
Point::Point(int dim, const vector<Ciphertext>& enc_coords) 
    : dim(dim), enc_coords(enc_coords) {
    coords.resize(dim);
}

Point::Point(int dim, const vector<float>& coords, const vector<Ciphertext>& enc_coords)
{
    this->dim = dim;
    this->coords = std::move(coords);
    this->enc_coords = std::move(enc_coords);
}

// 新构造函数：指定维度和坐标
Point::Point(const vector<float>& coords) 
   : coords(coords) {
    dim = coords.size();
    enc_coords.resize(dim);
}

// 新构造函数：指定维度和加密坐标
Point::Point(const vector<Ciphertext>& enc_coords) 
    :enc_coords(enc_coords) {
    dim = enc_coords.size();
    coords.resize(dim);
}

Point::Point(const vector<float>& coords, const vector<Ciphertext>& enc_coords)
{
    this->dim = coords.size();
    this->coords = std::move(coords);
    this->enc_coords = std::move(enc_coords);
}

// 保留但标记为过时
Point::Point(float x, float y) : dim(2) {
    coords = {x, y};
    enc_coords.resize(2);
}

void Point::encrypt_point(PaillierFast& paillier) {
    enc_coords.clear();
    for (int i = 0; i < dim; i++) {
        coords[i] = (coords[i] * 1000000) / 1000000;
        // cout<<coords[i]<<", ";
        // float scaled_val = std::round(coords[i] * 1000000) / 1000000;
        long long scaled_int = (coords[i] * 1000000);
        enc_coords.push_back(paillier.encrypt(Integer(scaled_int)));
    }
    // cout<<endl;
    // 其他字段加密保持不变
    temp_dist = (temp_dist * 1000000) / 1000000;
    // float scaled_temp = std::round(temp_dist * 1000000) / 1000000;
    long long scaled_temp_int = (temp_dist * 1000000);
    enc_temp_dist = paillier.encrypt(Integer(scaled_temp_int));
}

bool Point::operator==(const Point &other) const {
    if (dim != other.dim) return false;
    for (int i = 0; i < dim; i++) {
        if (coords[i] != other.coords[i]) return false;
    }
    return true;
}

float Point::cal_dist(const Point &other) const {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = coords[i] - other.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

Ciphertext Point::enc_cal_dist(const Point &other, PaillierFast& paillier) const {
    return SSEDrun(enc_coords, other.enc_coords, paillier);
}

void Point::print() {
    std::cout << "(";
    for (int i = 0; i < dim; i++) {
        std::cout << "dim" << i << "=" << coords[i];
        if (i < dim - 1) std::cout << ", ";
    }
    std::cout << ") index=" << index << " curve_val=" << curve_val << std::endl;
}

void Point::init() {
    for (int i = 0; i < dim; i++) {
        coords[i] = (coords[i] * 1000000) / 1000000;
    }
    temp_dist = (temp_dist * 1000000) / 1000000;
}

vector<Point> Point::get_points(vector<Point> dataset, int num) {
    srand(time(0));
    vector<Point> points;
    if (dataset.empty()) return points;
    int length = dataset.size();
    for (int i = 0; i < num; i++) {
        int index = rand() % length;
        points.push_back(dataset[index]);
    }
    return points;
}

vector<Point> Point::get_inserted_points(long long num, int dim) {
    srand(time(0));
    vector<Point> points;
    for (int i = 0; i < num; i++) {
        vector<float> coords;
        for (int d = 0; d < dim; d++) {
            coords.push_back(static_cast<float>(rand()) / RAND_MAX);
        }
        points.push_back(Point(dim, coords));
    }
    return points;
}

string Point::get_self() {
    string str = "";
    for (int i = 0; i < dim; i++) {
        str += to_string(coords[i]);
        if (i < dim - 1) str += ",";
    }
    return str + "\n";
}