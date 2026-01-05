#include <iostream>
#include <thread>
#include <vector>
#include "SM.h"
#include <gmp.h>
#include <gmpxx.h>
#include <string.h>
using namespace std;

// 线程 DSP的函数
void SSEDDSPFunction(std::vector<Ciphertext> x, std::vector<Ciphertext> y, Ciphertext& R, PaillierFast& paillier) 
{
    // 对两个向量每个维度执行相减操作得到 X 和 Y
    // int X = x[0] - y[0];
    // int Y = x[1] - y[1];
    if(x.size() != y.size())
    {
        cout<<"两个点的维度不匹配，无法计算欧式距离"<<endl;
    }
    R = paillier.encrypt(0);
    for(int i = 0; i < x.size(); i++)
    {
        Integer a = paillier.decrypt(x[i]);
        Integer b = paillier.decrypt(y[i]);
        Ciphertext z = x[i] -y[i];
        Integer c = paillier.decrypt(z);
        R = R + SMrun(z, z, paillier);
        Integer d = paillier.decrypt(R);
        cout<<"a: "<<a<<", b: "<<b<<", c: "<<c<<", d: "<<d<<endl;
    }
}

Ciphertext SSEDrun( std::vector<Ciphertext> x,std::vector<Ciphertext> y, PaillierFast& paillier) 
{
    //实验记录
    Ciphertext Result = paillier.encrypt(0);
    std::thread t1(SSEDDSPFunction, x, y, std::ref(Result), std::ref(paillier));
    // std::cout<<"SSEDDSP线程"<<std::endl;
    t1.join();
    Ciphertext final_result(Result);
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
//     vector<Ciphertext> X{A, C};
//     vector<Ciphertext> Y{B, D};
//     cout<<paillier.decrypt(SSEDrun(X, Y, paillier))<<endl;
//     return 0;
// }