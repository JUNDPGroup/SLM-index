// #ifndef NONLEAFNODE_H
// #define NONLEAFNODE_H

// #include <vector>
// #include "Node.h"
// // #include <boost/serialization/access.hpp>
// #include <boost/serialization/shared_ptr.hpp>
// #include <boost/serialization/weak_ptr.hpp>
// #include <boost/serialization/access.hpp>
// #include <boost/serialization/export.hpp>
// #include <boost/serialization/vector.hpp>
// #include "../utils/Constants.h"
// class NonLeafNode : public nodespace::Node
// {
//     Constants constant;
// public:
//     int level;
//     vector<Node*> *children;
//     // std::vector<Node*> children; // 改为对象直接持有
//     // NonLeafNode *parent = nullptr;
//     // std::vector<std::shared_ptr<Node>> children;
//     // std::weak_ptr<NonLeafNode> parent;

//     NonLeafNode();
//     NonLeafNode(Mbr mbr);

//     virtual void printInfo() {
//         std::cout << "This is a NonLeafNode." << std::endl;
//     }
//     // ~NonLeafNode(); // 新增析构函数
//     // 添加析构函数释放内存
//     // ~NonLeafNode() {
//     //     for (auto child : *children) delete child;
//     //     delete children;
//     // }
//     // virtual ~NonLeafNode() {}

//     void addNode(Node*);
//     void addNodes(vector<Node*>);
//     bool is_full();
//     NonLeafNode* split();

//     // 必须显式实现析构函数
//     ~NonLeafNode() {
//         for (Node* child : *children) {
//             delete child; // 递归释放子节点
//         }
//     }

//     // 序列化方法
//     friend class boost::serialization::access;
//     template<class Archive>
//     void serialize(Archive& ar, const unsigned int version) {
//         ar & boost::serialization::base_object<Node>(*this); // 序列化基类
//         ar & children; // 序列化子节点数组
//         // ar & parent;   // 序列化父指针
//         ar & mbr;
//     }
// };
// // BOOST_CLASS_EXPORT(NonLeafNode)  // 注册多态类
// // BOOST_CLASS_EXPORT_KEY(NonLeafNode)  // 声明多态类型
// #endif


#ifndef NONLEAFNODE_H
#define NONLEAFNODE_H

#include <vector>
#include "Node.h"
#include "../utils/Constants.h"
#include "../Serialize/serialization_helpers.hpp"
class NonLeafNode : public nodespace::Node
{
public:
    int level;
    vector<nodespace::Node*> *children;

    NonLeafNode();
    NonLeafNode(int dim);  // 新构造函数
    NonLeafNode(const Mbr& mbr);  // 修改为接受Mbr对象

    void addNode(nodespace::Node* node);
    void addNodes(vector<nodespace::Node*> nodes);
    bool is_full() const;
    NonLeafNode* split();

    ~NonLeafNode() {
        for (nodespace::Node* child : *children) {
            delete child;
        }
        delete children;
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & boost::serialization::base_object<Node>(*this);
        ar & children;
        ar & mbr;
    }
};

#endif