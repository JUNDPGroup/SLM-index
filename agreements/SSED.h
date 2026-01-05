#ifndef SSED_H
#define SSED_H

#include <iostream>
#include <thread>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>
#include <string.h>
#include "SM.h"
#include <ophelib/paillier_fast.h>
using namespace ophelib;
using namespace std;
// 假设 SM.h 已经有相关声明，如果这里有命名空间等需要额外处理
#include "SM.h"

// 线程 1 的函数声明
void SSEDDSPFunction(std::vector<Ciphertext> , std::vector<Ciphertext> , Ciphertext&, PaillierFast& );
Ciphertext SSEDrun(std::vector<Ciphertext>, std::vector<Ciphertext> , PaillierFast& );
#endif // SSED_H