#ifndef SM_H
#define SM_H

#include <iostream>
#include <thread>
#include <random>
// #include "../Paillier/paillier.h"
#include <gmp.h>
#include <gmpxx.h>
#include <string.h>
#include <ophelib/paillier_fast.h>
using namespace ophelib;
using namespace std;
// 随机数生成函数声明
int generateRandomNumber();

// 线程DAP的函数声明，接收 X 和 Y 作为参数，返回 Z
void SMDAPFunction(Ciphertext& , Ciphertext, Ciphertext, PaillierFast&);

// 线程DSP的函数声明
void SMDSPFunction(Ciphertext , Ciphertext , Ciphertext&, PaillierFast&);

// run 函数声明
Ciphertext SMrun(Ciphertext , Ciphertext, PaillierFast&);

#endif // YOUR_HEADER_H