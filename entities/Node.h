#ifndef NODE_H
#define NODE_H
#include "Mbr.h"
#include <boost/serialization/export.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/vector.hpp>
#include "../Serialize/serialization_helpers.hpp"
namespace nodespace
{
class Node
{
public:
    Mbr mbr;
    int order_in_level;
    Node();
    virtual ~Node() {}
    virtual void printInfo() {
        std::cout << "This is a Node." << std::endl;
    }
    float cal_dist(const Point& point) const ;
    // {
    //     return mbr.cal_dist(point);
    // }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & mbr & order_in_level;
    }
};
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(nodespace::Node)
#endif