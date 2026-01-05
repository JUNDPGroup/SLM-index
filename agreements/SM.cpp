#include <iostream>
#include <thread>
#include <random>
#include "SM.h"
// #include "../Paillier/paillier.h"
#include <gmp.h>
#include <gmpxx.h>
#include <string.h>
#include <ophelib/paillier_fast.h>
using namespace std;

// 随机数生成函数
int generateRandomNumber() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(1, 100);  // 生成 1 到 100 之间的随机数
    return dis(gen);
}

// 线程DAP的函数，接收 X 和 Y 作为参数，返回 Z
void SMDAPFunction(Ciphertext& Z, Ciphertext X, Ciphertext Y, PaillierFast& paillier) {
    //return X * Y;
    Integer x, y, z;
    x = paillier.decrypt(X);
    y = paillier.decrypt(Y);
    PublicKey pub = paillier.get_pub();
    z = x * y;
    z = z % pub.n;
    Z = paillier.encrypt(z);
}

// 线程DSP的函数
void SMDSPFunction(Ciphertext x, Ciphertext y, Ciphertext& R, PaillierFast& paillier) {
    // 随机生成 r1 和 r2
    int r1 = generateRandomNumber();
    int r2 = generateRandomNumber();


    Ciphertext X, Y, Z, s1, s2, enc_r1, enc_r2;
    Integer R1(r1), R2(r2);
    // 计算 X 和 Y
    // int X = x + r1;
    // int Y = y + r2;
    enc_r1 = paillier.encrypt(R1);
    enc_r2 = paillier.encrypt(R2);
    X = x + enc_r1;
    Y = y + enc_r2;
    std::thread t2([X, Y, &Z, &paillier]() {
        SMDAPFunction(Z, X, Y, paillier);
    });

    t2.join();



    // 计算 s1、s2 和 R
    // int s1 = Z - x * r2;
    // int s2 = s1 - y * r1;
    Ciphertext tmp, res;
    tmp = x * R2;
    s1 = Z - tmp;
    tmp = y * R1;
    s2 = s1 - tmp;

    int r = r1*r2;
    res = paillier.encrypt(Integer(r));
    R = s2 - res;
}

Ciphertext SMrun(Ciphertext x, Ciphertext y, PaillierFast& paillier) {

    //实验记录
    Ciphertext R;
    std::thread t1(SMDSPFunction, x, y, std::ref(R), std::ref(paillier));
    //std::cout<<"SMDSP线程"<<std::endl;
    t1.join();
    Ciphertext final_result = R;
    PublicKey pub = paillier.get_pub();
    // Integer n2 = pub.n * pub.n;
    // final_result = final_result % n2;
    return final_result;
}


// int main()
// {
//     Integer a(1), b(-1), c(2), d(-2), e(0);
//     Ciphertext A, B, C, D, E;
//     PaillierFast paillier(1024);
//     paillier.generate_keys();
//     A = paillier.encrypt(a);
//     B = paillier.encrypt(b);
//     C = paillier.encrypt(c);
//     D = paillier.encrypt(d);
//     E = paillier.encrypt(e);
//     cout<<paillier.decrypt(SMrun(A, B, paillier))<<endl;
//     cout<<paillier.decrypt(SMrun(C, D, paillier))<<endl;
//     cout<<paillier.decrypt(SMrun(D, D, paillier))<<endl;
//     cout<<paillier.decrypt(SMrun(C, E, paillier))<<endl;
//     cout<<paillier.decrypt(SMrun(E, A, paillier))<<endl;
//     cout<<paillier.decrypt(SMrun(D, D, paillier))<<endl;
//     return 0;
// }