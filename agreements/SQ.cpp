#include <iostream>
#include <thread>
#include <random>
#include "SQ.h"
// #include "../Paillier/paillier.h"
#include <gmp.h>
#include <gmpxx.h>
#include <string.h>
using namespace std;


// 随机数生成函数
int generateRandomNumber_1() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(1, 100);  // 生成 1 到 100 之间的随机数
    return dis(gen);
}


void SQDAPFunction(Integer& A, Integer& B, Ciphertext enc_1, Ciphertext enc_0, Ciphertext X, Ciphertext Y, PaillierFast& paillier)
{
    Integer one(1), zero(0);
    enc_1 = paillier.encrypt(one);
    enc_0 = paillier.encrypt(zero);
    A = paillier.decrypt(X);
    B = paillier.decrypt(Y);
    //cout<<A<<endl<<B<<endl;
}

void SQDSPFunction(Integer& A, Integer& B, Ciphertext enc_1, Ciphertext enc_0, Ciphertext x, Ciphertext y, PaillierFast& paillier)
{
    // 随机生成 r1 和 r2
    int r1 = generateRandomNumber_1();
    int r2 = generateRandomNumber_1();
    //X = x+r1, Y = y+r2
    Integer R1(r1), R2(r2);
    Ciphertext enc_r1, enc_r2, X, Y;
    enc_r1 = paillier.encrypt(R1);
    enc_r2 = paillier.encrypt(R2);
    X = x + enc_r1;
    Y = y + enc_r2;
    std::thread t2([&A, &B, enc_1, enc_0, X, Y, &paillier]() {
        SQDAPFunction(A, B, enc_1, enc_0, X, Y, paillier);
    });
    t2.join();

    A = A - r1;
    B = B - r2;

    //cout<<A<<endl<<B<<endl;
    
}

void SQrun(Integer& A, Integer& B, Ciphertext enc_1, Ciphertext enc_0, Ciphertext x, Ciphertext y, PaillierFast& paillier) 
{
    //mpz_class A, B;
    std::thread t1(SQDSPFunction, std::ref(A), std::ref(B), (enc_1), (enc_0), x, y, std::ref(paillier));
    t1.join();
    //cout<<A<<endl<<B<<endl;
}