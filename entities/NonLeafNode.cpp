// #include "NonLeafNode.h"
// #include "LeafNode.h"
// #include "../utils/Constants.h"
// #include "../Serialize/serialization_helpers.hpp" 
// #include <vector>
// #include <iostream>
// #include <typeinfo>
// using namespace std;
// extern Constants constants;
// NonLeafNode::NonLeafNode()
// {
//     children = new vector<Node *>();
//     // parent = nullptr;
// }

// NonLeafNode::NonLeafNode(Mbr mbr)
// {
//     children = new vector<Node *>();
//     // parent = nullptr;
//     this->mbr = mbr;
// }

// // NonLeafNode::~NonLeafNode() {
// //     for (auto* child : *children) delete child; // 释放子节点内存
// //     delete children;
// // }


// void NonLeafNode::addNode(Node *node)
// {
//     //assert(children->size() <= Constants::PAGESIZE);
//     // add
//     children->push_back(node);
//     // update MBR
//     mbr.update(node->mbr);

//     if (typeid(*node) == typeid(NonLeafNode))
//     {
//         NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(node);
//         // nonLeafNode->parent = this;
//     }
//     else
//     {
//         LeafNode *leafNode = dynamic_cast<LeafNode *>(node);
//         // leafNode->parent = this;
//     }
// }

// void NonLeafNode::addNodes(vector<Node *> nodes)
// {
//     for (int i = 0; i < nodes.size(); i++)
//     {
//         addNode(nodes[i]);
//     }
//     // cout<< mbr.x1 << " " << mbr.y1 << " " << mbr.x2 << " " << mbr.y2 << endl;
// }

// bool NonLeafNode::is_full()
// {
//     return children->size() >= constants.PAGESIZE;
// }

// NonLeafNode *NonLeafNode::split()
// {
//     // build rightNode
//     NonLeafNode *right = new NonLeafNode();
//     // right->parent = this->parent;
//     int mid = constants.PAGESIZE / 2;
//     auto bn = children->begin() + mid;
//     auto en = children->end();
//     vector<Node *> vec(bn, en);
//     right->addNodes(vec);
    
//     // build leftNode
//     auto bn1 = children->begin();
//     auto en1 = children->begin() + mid;
//     vector<Node *> vec1(bn1, en1);
//     children->clear();
//     addNodes(vec1);

//     return right;
// }

// //序列化时打开
// #include <boost/serialization/export.hpp>
// BOOST_CLASS_EXPORT_IMPLEMENT(NonLeafNode)


#include "NonLeafNode.h"
#include "LeafNode.h"
#include "../utils/Constants.h"
#include "../Serialize/serialization_helpers.hpp" 
#include <vector>
#include <iostream>
#include <typeinfo>
using namespace std;
extern Constants constants;

NonLeafNode::NonLeafNode()
{
    children = new vector<Node *>();
    mbr = Mbr(0); // 默认维度0
}

NonLeafNode::NonLeafNode(const Mbr &mbr)
{
    children = new vector<Node *>();
    this->mbr = mbr;
}

void NonLeafNode::addNode(Node *node)
{
    // 如果是第一个节点，初始化MBR维度
    if (mbr.dim == 0) {
        mbr = Mbr(node->mbr.dim);
    }
    // 检查维度一致性
    else if (node->mbr.dim != mbr.dim) {
        throw std::runtime_error("节点维度不匹配");
    }
    
    // 添加节点
    children->push_back(node);
    
    // 更新MBR
    mbr.update(node->mbr);
    
    // 更新父节点指针
    if (typeid(*node) == typeid(NonLeafNode)) {
        NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(node);
        // nonLeafNode->parent = this;
    } else {
        LeafNode *leafNode = dynamic_cast<LeafNode *>(node);
        // leafNode->parent = this;
    }
}

void NonLeafNode::addNodes(vector<Node *> nodes)
{
    for (int i = 0; i < nodes.size(); i++)
    {
        addNode(nodes[i]);
    }
}

bool NonLeafNode::is_full() const
{
    return children->size() >= constants.PAGESIZE;
}

NonLeafNode *NonLeafNode::split()
{
    // build rightNode
    NonLeafNode *right = new NonLeafNode(mbr);
    int mid = constants.PAGESIZE / 2;
    auto bn = children->begin() + mid;
    auto en = children->end();
    vector<Node *> vec(bn, en);
    right->addNodes(vec);
    
    // build leftNode
    auto bn1 = children->begin();
    auto en1 = children->begin() + mid;
    vector<Node *> vec1(bn1, en1);
    children->clear();
    addNodes(vec1);

    return right;
}

#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT_IMPLEMENT(NonLeafNode)