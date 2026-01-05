#include "ophelib/paillier_fast.h"
#include "ophelib/wire.h"
#include <fstream>
using namespace ophelib;
using namespace std;
namespace ophelib {

// 序列化PaillierFast
void serialize_paillier_fast(const PaillierFast& pai, const std::string& filename) {
    // 获取密钥对
    const KeyPair kp = pai.get_keypair();
    // 序列化到文件
    serialize_to_file(kp, filename);
}

// 反序列化PaillierFast
PaillierFast deserialize_paillier_fast(const std::string& filename) {
    // 反序列化密钥对
    KeyPair kp = deserialize_from_file<KeyPair>(filename);
    // 重建PaillierFast对象
    if (kp.priv.key_size_bits > 0) {
        return PaillierFast(kp.pub, kp.priv);
    } else {
        return PaillierFast(kp.pub);
    }
}

// 序列化Integer
void serialize_integer(const Integer& num, const std::string& filename) {
    serialize_to_file(num, filename);
}

// 反序列化Integer
Integer deserialize_integer(const std::string& filename) {
    return deserialize_from_file<Integer>(filename);
}

// 序列化Ciphertext
void serialize_ciphertext(const Ciphertext& cipher, const std::string& filename) {
    serialize_to_file(cipher, filename);
}

// 反序列化Ciphertext (需要n²模数共享指针)
Ciphertext deserialize_ciphertext(
    const std::string& filename, 
    const std::shared_ptr<Integer>& n2_shared
) {
    Ciphertext cipher = deserialize_from_file<Ciphertext>(filename);
    cipher.n2_shared = n2_shared;
    return cipher;
}

} // namespace ophelib
