// Fully extended Mbr class for n-dimensional support (header-compatible)
#include "Mbr.h"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <gmp.h>
#include <gmpxx.h>
#include "../agreements/SIC.h"
#include "../agreements/SM.h"
#include "../agreements/SSED.h"
#include "../Serialize/serialization_helpers.hpp"

using namespace std;

Mbr::Mbr() : dim(0) 
{
}

Mbr::Mbr(int dim) : dim(dim) 
{
    lows = vector<float>(dim);
    highs = vector<float>(dim);
    enc_lows = vector<Ciphertext>(dim);
    enc_highs = vector<Ciphertext>(dim);
    float x = numeric_limits<float>::max();
    float y = numeric_limits<float>::min();
    for(int i = 0; i < dim; i++)
    {
        lows[i] = x;
        highs[i] = y;
    }
}

Mbr::Mbr(int dim, const vector<float>& lows, const vector<float>& highs) : dim(dim), lows(lows), highs(highs) 
{
    // for(int i = 0; i < dim; i++)
    // {
    //     cout<<"lows: "<<lows[i]<<", highs: "<<highs[i]<<endl;
    // }
    enc_lows = vector<Ciphertext>(dim);
    enc_highs = vector<Ciphertext>(dim);
}

void Mbr::init() {
    for (int i = 0; i < dim; ++i) {
        lows[i] = (lows[i] * 1000000) / 1000000;
        highs[i] = (highs[i] * 1000000) / 1000000;
    }
}

void Mbr::clear()
{
    float x = numeric_limits<float>::max();
    float y = numeric_limits<float>::min();
    for(int i = 0; i < dim; i++)
    {
        lows[i] = x;
        highs[i] = y;
    }
}

void Mbr::encrypt_mbr(PaillierFast& paillier) {
    // cout<<"加密Mbr"<<endl;
    enc_lows.clear();
    enc_highs.clear();
    for (int i = 0; i < dim; ++i) {
        lows[i] = (lows[i] * 1000000) / 1000000;
        highs[i] = (highs[i] * 1000000) / 1000000;
        long long scaled_low = (lows[i] * 1000000);
        long long scaled_high = (highs[i] * 1000000);
        // cout<<"lows: "<<lows[i]<<" highs: "<<highs[i]<<endl;
        // cout<<"lows: "<<scaled_low<<" highs: "<<scaled_high<<endl;
        enc_lows.push_back(paillier.encrypt(Integer(scaled_low)));
        enc_highs.push_back(paillier.encrypt(Integer(scaled_high)));
    }
}

void Mbr::update(const Point& point) {
    for (int i = 0; i < dim; ++i) {
        lows[i] = min(lows[i], point.coords[i]);
        highs[i] = max(highs[i], point.coords[i]);
    }
}

void Mbr::update(const Point& point, PaillierFast& paillier) {
    update(point);
    enc_update(point.enc_coords, paillier);
}

void Mbr::enc_update(const vector<Ciphertext>& enc_coords, PaillierFast& paillier) {
    for (int i = 0; i < dim; ++i) {
        if (SICrun(enc_lows[i], enc_coords[i], paillier) == 0) enc_lows[i] = enc_coords[i];
        if (SICrun(enc_coords[i], enc_highs[i], paillier) == 0) enc_highs[i] = enc_coords[i];
    }
}

void Mbr::update(const Mbr& other) {
    for (int i = 0; i < dim; ++i) {
        lows[i] = min(lows[i], other.lows[i]);
        highs[i] = max(highs[i], other.highs[i]);
    }
}

bool Mbr::contains(const Point& point) const {
    for (int i = 0; i < dim; ++i) {
        // 将浮点数比较改为扩大1e6后的整数比较
        long long pointCoord = static_cast<long long>(point.coords[i] * 1000000);
        long long lowVal = static_cast<long long>(lows[i] * 1000000);
        long long highVal = static_cast<long long>(highs[i] * 1000000);
        // cout<<"pointCoord: "<<pointCoord<<" lowVal: "<<lowVal<<" highVal"<<highVal<<endl;
        if (pointCoord < lowVal || pointCoord > highVal) return false;
    }
    return true;
}


bool Mbr::enc_contains(const Point& point, PaillierFast& paillier) const {
    for (int i = 0; i < dim; ++i) {
        // cout<<"point: "<<paillier.decrypt(point.enc_coords[i])<<", lows: "<<paillier.decrypt(enc_lows[i])<<", highs: "<<paillier.decrypt(enc_highs[i])<<endl;

        if (SICrun(enc_lows[i], point.enc_coords[i], paillier) == 0 || SICrun(point.enc_coords[i], enc_highs[i], paillier) == 0)
            return false;
    }
    return true;
}

bool Mbr::strict_contains(const Point& point) const {
    for (int i = 0; i < dim; ++i) {
        if (!(lows[i] < point.coords[i] && point.coords[i] < highs[i])) return false;
    }
    return true;
}

bool Mbr::enc_strict_contains(const Point& point, PaillierFast& paillier) const {
    for (int i = 0; i < dim; ++i) {
        if (!(SICrun(point.enc_coords[i], enc_lows[i], paillier) == 0 && SICrun(enc_highs[i], point.enc_coords[i], paillier) == 0))
            return false;
    }
    return true;
}

bool Mbr::interact(const Mbr& other) const {
    for (int i = 0; i < dim; ++i) {
        // 将浮点数比较改为扩大1e6后的整数比较
        long long thisHigh = static_cast<long long>(highs[i] * 1000000);
        long long otherLow = static_cast<long long>(other.lows[i] * 1000000);
        long long otherHigh = static_cast<long long>(other.highs[i] * 1000000);
        long long thisLow = static_cast<long long>(lows[i] * 1000000);
        // cout<<"thisHigh: "<<thisHigh<<" "<<"thisLow: "<<thisLow<<"\n"<<"otherHigh: "<<otherHigh<<" "<< "otherLow: "<<otherLow<<endl;
        // cout<<"thisHigh: "<<highs[i]<<" "<<"thisLow: "<<lows[i]<<"\n"<<"otherHigh: "<<other.highs[i]<<" "<< "otherLow: "<<other.lows[i]<<endl;
        if (thisHigh < otherLow || otherHigh < thisLow) return false;
    }
    return true;
}

bool Mbr::enc_interact(const Mbr& other, PaillierFast& paillier) const {
    for (int i = 0; i < dim; ++i) {
        // cout<<"other_low: "<<other.lows[i]<<", low: "<<lows[i]<<endl;
        // cout<<"other_high: "<<other.highs[i]<<", high: "<<highs[i]<<endl;
        if (SICrun(other.enc_lows[i], enc_highs[i], paillier) == 0 || SICrun(enc_lows[i], other.enc_highs[i], paillier) == 0)
            return false;
    }
    return true;
}

// vector<Mbr> Mbr::get_mbrs(vector<Point> dataset, float area, int num, float ratio) {
//     vector<Mbr> mbrs;
//     srand(time(0));
//     int length = dataset.size();
//     int dim = dataset[0].coords.size();
//     vector<float> sizes(dim);
//     sizes[0] = sqrt(area * ratio);
//     if (dim > 1) sizes[1] = sqrt(area / ratio);
//     for (int i = 2; i < dim; ++i) sizes[i] = 0.1f;

//     int i = 0;
//     while (i < num) {
//         int index = rand() % length;
//         const auto& point = dataset[index];
//         vector<float> l(dim), h(dim);
//         bool valid = true;
//         for (int j = 0; j < dim; ++j) {
//             l[j] = point.coords[j];
//             h[j] = point.coords[j] + sizes[j];
//             // cout<<"l: "<<l[j]<<" h: "<<h[j]<<endl;
//             if (h[j] > 1.0f) valid = false;
//         }
//         if (valid) {
//             Mbr mbr(dim, l, h);
//             // mbrs.emplace_back(dim, l, h);
//             mbrs.push_back(mbr);
//             ++i;
//         }
//     }
//     return mbrs;
// }

// vector<Mbr> Mbr::get_mbrs(vector<Point> dataset, float area, int num, vector<float> ratios) {
//     vector<Mbr> mbrs;
//     // srand(time(0));

//     int length = dataset.size();
//     int dim = dataset[0].coords.size();
//     if ((int)ratios.size() != dim) {
//         throw std::invalid_argument("ratios size must equal dimension of dataset points");
//     }

//     // === 计算每个维度的长度 ===
//     float prod_ratio = 1.0f;
//     for (float r : ratios) prod_ratio *= r;

//     float scale = pow(area / prod_ratio, 1.0f / dim);

//     vector<float> sizes(dim);
//     for (int j = 0; j < dim; ++j) {
//         sizes[j] = ratios[j] * scale;
//     }

//     // === 随机生成 MBR ===
//     int i = 0;
//     while (i < num) {
//         int index = rand() % length;
//         const auto& point = dataset[index];

//         vector<float> l(dim), h(dim);
//         bool valid = true;

//         for (int j = 0; j < dim; ++j) {
//             l[j] = point.coords[j];
//             h[j] = point.coords[j] + sizes[j];
//             // cout<<"l: "<<l[j]<<" h: "<<h[j]<<endl;
//             if (h[j] > 1.0f) valid = false; // 保证不越界
//         }

//         if (valid) {
//             mbrs.emplace_back(dim, l, h);
//             ++i;
//         }
//     }

//     return mbrs;
// }

vector<Mbr> Mbr::get_mbrs(const vector<Point>& dataset, float selectivity, int num) {
    vector<Mbr> mbrs;
    // 输入合法性校验
    if (dataset.empty() || num <= 0 || selectivity <= 0 || selectivity >= 1.0f) {
        return mbrs;
    }

    srand(time(0));
    int totalPoints = dataset.size();
    int dim = dataset[0].dim; // 从Point类的dim成员获取维度
    
    // 验证所有点维度一致
    for (const auto& p : dataset) {
        if (p.dim != dim) {
            throw invalid_argument("All points must have the same dimension");
        }
    }

    // 计算目标包含点数（选择率 × 总点数）
    int targetCount = static_cast<int>(totalPoints * selectivity);
    targetCount = max(1, targetCount); // 至少包含1个点

    // 预计算各维度的数据范围（用于初始化搜索半径）
    vector<float> minDim(dim, numeric_limits<float>::max());
    vector<float> maxDim(dim, numeric_limits<float>::min());
    for (const auto& p : dataset) {
        for (int i = 0; i < dim; ++i) {
            // 假设Point通过[]运算符访问第i维坐标（如p[i]）
            float val = p.coords[i]; 
            minDim[i] = min(minDim[i], val);
            maxDim[i] = max(maxDim[i], val);
        }
    }

    int generated = 0;
    while (generated < num) {
        // 1. 随机选择基准点
        int baseIdx = rand() % totalPoints;
        const Point& basePoint = dataset[baseIdx];

        // 2. 初始化各维度搜索半径（数据范围的1%）
        vector<float> radius(dim);
        for (int i = 0; i < dim; ++i) {
            radius[i] = (maxDim[i] - minDim[i]) * 0.01f;
        }

        // 3. 自适应调整半径以匹配目标点数
        for (int iter = 0; iter < 200; ++iter) {
            vector<Point> inRange;
            for (const auto& p : dataset) {
                bool inside = true;
                for (int i = 0; i < dim; ++i) {
                    // 访问Point的第i维坐标（根据实际成员调整，此处用p[i]示例）
                    if (p.coords[i] < basePoint.coords[i] - radius[i] || p.coords[i] > basePoint.coords[i] + radius[i]) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    inRange.push_back(p);
                }
            }

            // 4. 检查是否符合目标，生成MBR
            if (inRange.size() >= targetCount * 0.9 && inRange.size() <= targetCount * 1.1) {
                cout<<"inRange: "<<inRange.size()<<endl;
                vector<float> mbrMin(dim, numeric_limits<float>::max());
                vector<float> mbrMax(dim, numeric_limits<float>::min());
                for (const auto& p : inRange) {
                    for (int i = 0; i < dim; ++i) {
                        mbrMin[i] = min(mbrMin[i], p.coords[i]); // 取第i维最小值
                        mbrMax[i] = max(mbrMax[i], p.coords[i]); // 取第i维最大值
                    }
                }
                Mbr mbr(dim, mbrMin, mbrMax);
                mbrs.push_back(mbr); // 构造MBR（假设Mbr支持vector<float>参数）
                // mbr.print();
                generated++;
                break;
            } else if (inRange.size() < targetCount) {
                // 扩大半径
                for (int i = 0; i < dim; ++i) radius[i] *= 1.2f;
            } else {
                // 缩小半径
                for (int i = 0; i < dim; ++i) radius[i] *= 0.8f;
            }
        }
    }

    return mbrs;
}



float Mbr::cal_dist(const Point& point) const {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = 0.0f;
        if (point.coords[i] < lows[i]) {
            diff = lows[i] - point.coords[i];
        } else if (point.coords[i] > highs[i]) {
            diff = point.coords[i] - highs[i];
        }
        dist += diff * diff;
    }
    return sqrt(dist);
}


Ciphertext Mbr::enc_cal_dist(const Point& point, PaillierFast& paillier) const {
    Ciphertext enc_dist = paillier.encrypt(Integer(0));
    for (int i = 0; i < dim; ++i) {
        Ciphertext diff;
        // 如果 point[i] < low[i]
        if (SICrun(enc_lows[i], point.enc_coords[i], paillier) == 0) {
            diff = enc_lows[i] - point.enc_coords[i];
            diff = SMrun(diff, diff, paillier);
            enc_dist += diff;
        }
        // 如果 point[i] > high[i]
        else if (SICrun(point.enc_coords[i], enc_highs[i], paillier) == 0) {
            diff = point.enc_coords[i] - enc_highs[i];
            diff = SMrun(diff, diff, paillier);
            enc_dist += diff;
        }
        // 如果 point[i] 在区间 [low, high] 内则不做处理
    }
    return enc_dist;
}


void Mbr::print() {
    cout << "[";
    for (int i = 0; i < dim; ++i) {
        cout << "(" << lows[i] << ", " << highs[i] << ")";
        if (i != dim - 1) cout << ", ";
    }
    cout << "]" << endl;
}

vector<Point> Mbr::get_corner_points() const {
    vector<Point> result;
    for (int i = 0; i < (1 << dim); ++i) {
        vector<float> coords(dim);
        for (int j = 0; j < dim; ++j)
            coords[j] = (i & (1 << j)) ? highs[j] : lows[j];
        result.emplace_back(coords);
    }
    return result;
}

vector<Point> Mbr::enc_get_corner_points() const {
    vector<Point> result;
    for (int i = 0; i < (1 << dim); ++i) {
        vector<Ciphertext> coords(dim);
        for (int j = 0; j < dim; ++j)
            coords[j] = (i & (1 << j)) ? enc_highs[j] : enc_lows[j];
        result.emplace_back(coords);
    }
    return result;
}

vector<Point> Mbr::init_get_corner_points() {
    vector<Point> result;
    for (int i = 0; i < (1 << dim); ++i) {
        vector<float> p(dim);
        vector<Ciphertext> enc_p(dim);
        for (int j = 0; j < dim; ++j) {
            if (i & (1 << j)) {
                p[j] = highs[j];
                enc_p[j] = enc_highs[j];
            } else {
                p[j] = lows[j];
                enc_p[j] = enc_lows[j];
            }
        }
        result.emplace_back(p, enc_p);
    }
    return result;
}

Mbr Mbr::get_mbr(const Point& point, float knnquerySide, int dim) {
    vector<float> l(dim), h(dim);
    for (int i = 0; i < dim; ++i) {
        l[i] = max(0.0f, point.coords[i] - knnquerySide);
        h[i] = min(1.0f, point.coords[i] + knnquerySide);
    }
    return Mbr(dim, l, h);
}

void Mbr::clean() {
    fill(lows.begin(), lows.end(), 0.0f);
    fill(highs.begin(), highs.end(), 0.0f);
}

string Mbr::get_self() {
    string result;
    for (int i = 0; i < dim; ++i) {
        result += to_string(lows[i]);
        result += ",";
    }
    for (int i = 0; i < dim; ++i) {
        result += to_string(highs[i]);
        if (i != dim - 1) result += ",";
    }
    result += "\n";
    return result;
}
