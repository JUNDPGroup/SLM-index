#ifndef __PAILLIER_H
#define __PAILLIER_H

#include <iostream>
#include <gmp.h>
#include <gmpxx.h>
#include <omp.h>
#include <fstream>    // 添加文件流操作支持
#include <string>     // 确保字符串操作支持
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
class Paillier
{
public:
    // 成员函数声明
    mpz_t p, q, g, n, nsquare; // nsquare = n^2
    mpz_t lambda, lmdInv;      // lmdInv = lambda^{-1} mod n
    // mpz_class enc_zero, enc_one;
    // static mpz_class enc_zero; // 加密的零值
    Paillier(){
        mpz_inits(p, q, g, n, nsquare, lambda, lmdInv, NULL);
        // Encrypt(enc_zero.get_mpz_t(), mpz_class(0).get_mpz_t());
        // Encrypt(enc_one.get_mpz_t(), mpz_class(1).get_mpz_t());
    }
    ~Paillier(){
        mpz_clears(p, q, g, n, nsquare, lambda, lmdInv, NULL);
        std::cout << "释放内存并推出程序......" << std::endl;
    }

    // 拷贝赋值运算符
    Paillier& operator=(const Paillier& other) {
        if (this != &other) {
            // 实现拷贝赋值逻辑
        }
        return *this;
    }

    // 移动赋值运算符
    Paillier& operator=(Paillier&& other) noexcept {
        if (this != &other) {
            // 实现移动赋值逻辑
        }
        return *this;
    }

    // void enc_clear();

    // 序列化接口
    void save(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        boost::archive::text_oarchive oa(ofs);
        oa << *this;
    }

    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        boost::archive::text_iarchive ia(ifs);
        ia >> *this;
    }

    void KeyGen(unsigned long bitLen);
    void Encrypt(mpz_t c, mpz_t m); // 加密
    void Decrypt(mpz_t m, mpz_t c); // 解密
    void Add(mpz_t res, mpz_t c1, mpz_t c2);
    void Sub(mpz_t res, mpz_t a, mpz_t b);
    void Mul(mpz_t resc, mpz_t c, mpz_t e);
    // mpz_class Enc_float(float x, Paillier& paillier);
    


private:
    // 序列化辅助函数
    friend class boost::serialization::access;
    
    template<class Archive>
    void save(Archive& ar, const unsigned int version) const {
        std::vector<std::string> mpz_strings;
        
        // 将每个mpz_t转换为字符串
        mpz_strings.push_back(mpz_get_str(NULL, 10, p));
        mpz_strings.push_back(mpz_get_str(NULL, 10, q));
        mpz_strings.push_back(mpz_get_str(NULL, 10, g));
        mpz_strings.push_back(mpz_get_str(NULL, 10, n));
        mpz_strings.push_back(mpz_get_str(NULL, 10, nsquare));
        mpz_strings.push_back(mpz_get_str(NULL, 10, lambda));
        mpz_strings.push_back(mpz_get_str(NULL, 10, lmdInv));
        
        ar & mpz_strings;
    }

    template<class Archive>
    void load(Archive& ar, const unsigned int version) {
        std::vector<std::string> mpz_strings;
        ar & mpz_strings;
        
        // 确保数据完整性
        if(mpz_strings.size() != 7) {
            throw std::runtime_error("Invalid Paillier serialization data");
        }
        
        // 初始化并设置值
        mpz_set_str(p, mpz_strings[0].c_str(), 10);
        mpz_set_str(q, mpz_strings[1].c_str(), 10);
        mpz_set_str(g, mpz_strings[2].c_str(), 10);
        mpz_set_str(n, mpz_strings[3].c_str(), 10);
        mpz_set_str(nsquare, mpz_strings[4].c_str(), 10);
        mpz_set_str(lambda, mpz_strings[5].c_str(), 10);
        mpz_set_str(lmdInv, mpz_strings[6].c_str(), 10);
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

};

namespace boost {
    namespace serialization {
    
    template<class Archive>
    void save(Archive& ar, const mpz_t& gmp_num, const unsigned int version) {
        char* str = mpz_get_str(NULL, 10, gmp_num);
        ar & std::string(str);
        free(str);
    }
    
    template<class Archive>
    void load(Archive& ar, mpz_t& gmp_num, const unsigned int version) {
        std::string str;
        ar & str;
        mpz_init(gmp_num);
        mpz_set_str(gmp_num, str.c_str(), 10);
    }
    
    template<class Archive, class T, class U>
    void save(Archive& ar, const __gmp_expr<T, U>& expr, const unsigned int version) {
        mpz_class temp(expr);  // 通过mpz_class中转
        std::string s = temp.get_str();
        ar & s;
    }
    
    template<class Archive, class T, class U>
    void load(Archive& ar, __gmp_expr<T, U>& expr, const unsigned int version) {
        std::string s;
        ar & s;
        mpz_class temp(s);
        expr = __gmp_expr<T, U>(temp.get_mpz_t()); // 正确构造GMP表达式
    }
    
    template<class Archive, class T, class U>
    void serialize(Archive& ar, __gmp_expr<T, U>& expr, const unsigned int version) {
        split_free(ar, expr, version);
    }
    
    } // namespace serialization
    } // namespace boost
    
#endif