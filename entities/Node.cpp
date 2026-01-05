#include "Node.h"
#include "../Serialize/serialization_helpers.hpp"
nodespace::Node::Node()
{
}

float nodespace::Node::cal_dist(const Point &point) const
{
    return mbr.cal_dist(point);
}