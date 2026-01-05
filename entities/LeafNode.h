#ifndef LEAFNODE_H
#define LEAFNODE_H

#include <vector>
#include <memory>
#include "Node.h"
#include "Point.h"
#include "Mbr.h"
#include <ophelib/paillier_fast.h>
#include "../Serialize/serialization_helpers.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_member.hpp>

using namespace ophelib;
using namespace std;

class LeafNode : public nodespace::Node
{
public:
    int level;
    int dim;
    shared_ptr<vector<Point>> children;

    LeafNode();
    LeafNode(int dim);
    LeafNode(const Mbr& mbr);

    // 声明拷贝构造函数
    LeafNode(const LeafNode& other);
    // 声明赋值运算符（遵循“三法则”，确保拷贝后资源独立）
    LeafNode& operator=(const LeafNode& other);

    void encrypt_leaf(PaillierFast& paillier);
    void add_point(const Point& point);
    void enc_add_point(const Point& point, PaillierFast& paillier);
    void add_points(const vector<Point>& points);
    void enc_add_points(const vector<Point>& points, PaillierFast& paillier);
    bool is_full() const;
    LeafNode *split();
    LeafNode split1();
    bool delete_point(const Point& point);
    bool enc_delete_point(const Point& point, PaillierFast& paillier);

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        boost::serialization::split_member(ar, *this, version);
    }

    template <class Archive>
    void save(Archive &ar, const unsigned int version) const
    {
        ar & boost::serialization::base_object<Node>(*this);
        ar & level;
        ar & dim;
        ar & mbr;
        ar & children;
    }

    template <class Archive>
    void load(Archive &ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<Node>(*this);
        ar & level;
        ar & dim;
        ar & mbr;
        ar & children;
    }
};

#endif
