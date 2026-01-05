// #include "NodeExtend.h"
// #include <iostream>
// #include "NonLeafNode.h"
// #include <typeinfo>
// #include "../Serialize/serialization_helpers.hpp" 
// using namespace std;

// NodeExtend::NodeExtend()
// {
// }

// NodeExtend::NodeExtend(Point point, float dist)
// {
//     this->point = point;
//     this->dist = dist;
// }

// NodeExtend::NodeExtend(nodespace::Node *node, float dist)
// {
//     this->node = node;
//     this->dist = dist;
// }

// bool NodeExtend::is_leafnode()
// {
//     if (typeid(*node) == typeid(NonLeafNode))
//     {
//         return false;
//     }
//     return true;
// }
#include "NodeExtend.h"
#include "NonLeafNode.h"
#include <typeinfo>
#include "../Serialize/serialization_helpers.hpp"
NodeExtend::NodeExtend() : dist(0.0f) {}

NodeExtend::NodeExtend(nodespace::Node* node, float dist) 
    : node(node), dist(dist) {}

NodeExtend::NodeExtend(const Point& point, float dist) 
    : point(point), dist(dist) {}

bool NodeExtend::is_leafnode() const {
    return (dynamic_cast<NonLeafNode*>(node) == nullptr);
}