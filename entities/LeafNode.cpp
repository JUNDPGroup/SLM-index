#include <iostream>
#include "Node.h"
#include "LeafNode.h"
#include "Point.h"
#include "../utils/Constants.h"
#include "../Paillier/paillier.h"
#include "../Serialize/serialization_helpers.hpp" 
#include <algorithm>
#include <gmpxx.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <stdexcept>
#include <memory>

using namespace std;
extern Constants constants;

LeafNode::LeafNode() {
    children = std::make_shared<std::vector<Point>>();
    mbr = Mbr(0);
    dim = 0;
}

LeafNode::LeafNode(int dim) {
    children = std::make_shared<std::vector<Point>>();
    this->dim = dim;
    mbr = Mbr(dim);
}

LeafNode::LeafNode(const Mbr& mbr) {
    this->dim = mbr.dim;
    this->mbr = mbr;
    children = std::make_shared<std::vector<Point>>();
}

// 拷贝构造函数：深拷贝所有成员，包括 children 指向的 vector<Point>
LeafNode::LeafNode(const LeafNode& other)
    : Node(other)  // 拷贝基类 Node 的成员（如 mbr）
    , level(other.level)
    , dim(other.dim)
    // , mbr(other.mbr)  // Mbr 本身的拷贝会处理其内部 vector（深拷贝）
{
    this->mbr = other.mbr;
    // 关键：深拷贝 children 指向的 vector<Point>
    if (other.children != nullptr) {
        // 创建新的 vector<Point>，并复制原 vector 中的所有元素
        this->children = std::make_shared<std::vector<Point>>(*(other.children));
    } else {
        // 若原 children 为空，新 children 也置空
        this->children = nullptr;
    }
}

// 赋值运算符：先拷贝一份临时对象，再与当前对象交换资源
LeafNode& LeafNode::operator=(const LeafNode& other)
{
    if (this != &other)  // 避免自赋值（自己赋值给自己时无需操作）
    {
        // 1. 拷贝 other 到临时对象（触发上面定义的拷贝构造函数，完成深拷贝）
        LeafNode temp(other);
        // 2. 交换当前对象和临时对象的所有成员（包括 children）
        std::swap(level, temp.level);
        std::swap(dim, temp.dim);
        std::swap(mbr, temp.mbr);
        std::swap(children, temp.children);
        // 3. 临时对象 temp 离开作用域时，自动释放原对象的旧资源
    }
    return *this;
}

void LeafNode::encrypt_leaf(PaillierFast& paillier) {
    // std::cout << "加密叶节点，维度 = " << mbr.dim << std::endl;
    for (auto& p : *children) {
        if (p.get_dim() != mbr.dim) {
            throw std::runtime_error("点维度不匹配");
        }
        p.encrypt_point(paillier);
    }
    mbr.encrypt_mbr(paillier);
}

void LeafNode::add_point(const Point& point) {
    if (mbr.dim == 0) {
        mbr = Mbr(point.get_dim());
    } else if (point.get_dim() != mbr.dim) {
        throw std::runtime_error("点维度不匹配");
    }
    children->push_back(point);
    mbr.update(point);
}

void LeafNode::enc_add_point(const Point& point, PaillierFast& paillier) {
    if (mbr.dim == 0) {
        mbr = Mbr(point.get_dim());
    } else if (point.get_dim() != mbr.dim) {
        throw std::runtime_error("点维度不匹配");
    }
    children->push_back(point);
    mbr.update(point, paillier);
}

void LeafNode::add_points(const std::vector<Point>& points) {
    for (const auto& p : points) {
        add_point(p);
    }
}

void LeafNode::enc_add_points(const std::vector<Point>& points, PaillierFast& paillier) {
    for (const auto& p : points) {
        enc_add_point(p, paillier);
    }
}

bool LeafNode::is_full() const {
    return children->size() >= constants.PAGESIZE;
}

LeafNode* LeafNode::split() {
    LeafNode* right = new LeafNode(mbr);
    int mid = constants.PAGESIZE / 2;
    std::vector<Point> vec(children->begin() + mid, children->end());
    right->add_points(vec);

    std::vector<Point> vec1(children->begin(), children->begin() + mid);
    children->clear();
    add_points(vec1);
    return right;
}

LeafNode LeafNode::split1() {
    LeafNode right(mbr);
    int mid = constants.PAGESIZE / 2;
    std::vector<Point> vec(children->begin() + mid, children->end());
    right.add_points(vec);

    right.mbr.enc_lows = this->mbr.enc_lows;
    right.mbr.enc_highs = this->mbr.enc_highs;

    std::vector<Point> vec1(children->begin(), children->begin() + mid);
    children->clear();
    add_points(vec1);
    return right;
}

bool LeafNode::delete_point(const Point& point) {
    auto iter = std::find(children->begin(), children->end(), point);
    if (iter != children->end()) {
        children->erase(iter);
        if (!mbr.strict_contains(point)) {
            mbr.clean();
            for (const auto& p : *children) {
                mbr.update(p);
            }
        }
        return true;
    }
    return false;
}

bool LeafNode::enc_delete_point(const Point& point, PaillierFast& paillier) {
    auto iter = std::find(children->begin(), children->end(), point);
    if (iter != children->end()) {
        children->erase(iter);
        if (!mbr.enc_strict_contains(point, paillier)) {
            mbr.clean();
            for (const auto& p : *children) {
                mbr.enc_update(p.enc_coords, paillier);
            }
        }
        return true;
    }
    return false;
}

#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT_IMPLEMENT(LeafNode)
