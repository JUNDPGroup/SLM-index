#ifndef SERIALIZATION_HELPERS_HPP
#define SERIALIZATION_HELPERS_HPP

#include <boost/serialization/split_free.hpp>
#include <ophelib/paillier_fast.h>
#include <ophelib/integer.h>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace boost {
namespace serialization {

// Integer序列化支持
template<class Archive>
void save(Archive& ar, const ophelib::Integer& g, const unsigned int version) {
    std::string s = g.to_string_(10);  // 使用十进制
    ar & s;
}

template<class Archive>
void load(Archive& ar, ophelib::Integer& g, const unsigned int version) {
    std::string s;
    ar & s;
    g = ophelib::Integer(s.c_str(), 10);  // 从十进制字符串构造
}

// Ciphertext序列化支持
template<class Archive>
void save(Archive& ar, const ophelib::Ciphertext& c, const unsigned int version) {
    // 序列化数据部分
    ar & c.data;
    
    // 序列化n2_shared指针（如果存在）
    bool has_n2_shared = (c.n2_shared != nullptr);
    ar & has_n2_shared;
    
    if (has_n2_shared) {
        ar & *c.n2_shared;
    }
}

template<class Archive>
void load(Archive& ar, ophelib::Ciphertext& c, const unsigned int version) {
    // 加载数据部分
    ar & c.data;
    
    // 加载n2_shared指针
    bool has_n2_shared;
    ar & has_n2_shared;
    
    if (has_n2_shared) {
        c.n2_shared = std::make_shared<ophelib::Integer>();
        ar & *c.n2_shared;
    } else {
        c.n2_shared = nullptr;
    }
    
    // 注意：fast_mod 需要从Paillier对象重建
    c.fast_mod = nullptr;
}

// PublicKey序列化支持
template<class Archive>
void serialize(Archive& ar, ophelib::PublicKey& pub, const unsigned int version) {
    ar & pub.key_size_bits;
    ar & pub.n;
    ar & pub.g;
}

// PrivateKey序列化支持
template<class Archive>
void serialize(Archive& ar, ophelib::PrivateKey& priv, const unsigned int version) {
    ar & priv.key_size_bits;
    ar & priv.a_bits;
    ar & priv.p;
    ar & priv.q;
    ar & priv.a;
}

// KeyPair序列化支持
template<class Archive>
void serialize(Archive& ar, ophelib::KeyPair& kp, const unsigned int version) {
    ar & kp.pub;
    ar & kp.priv;
}

} // namespace serialization
} // namespace boost

// 注册序列化函数
BOOST_SERIALIZATION_SPLIT_FREE(ophelib::Integer)

// 为Ciphertext添加serialize适配器
namespace boost {
namespace serialization {
    template<class Archive>
    void serialize(Archive & ar, ophelib::Ciphertext & c, const unsigned int version) {
        split_free(ar, c, version);
    }
}
} // namespace boost

#endif // SERIALIZATION_HELPERS_HPP