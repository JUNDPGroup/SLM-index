#include "paillier.h"
#include <iostream>
#include <gmp.h>

using namespace std;

// int main()
// {
//     cout << "start ......" << endl;
//     Paillier paillier;
//     unsigned long bitLen = 1024;
//     paillier.KeyGen(bitLen);

//     mpz_t m1, m2, res, c1, c2, mm, x;
//     mpz_inits(m1, m2, res, c1, c2, mm, x, NULL);
//     mpz_set_ui(m1, 400000000);
//     mpz_set_ui(m2, 500000000);

//     paillier.Encrypt(c1, m1);
//     paillier.Encrypt(c2, m2);

//     string str1,str2,str3,str4,str5;
//     str1 = mpz_get_str(nullptr,10,c1);

//     str2 = mpz_get_str(nullptr,10,c2);

//     paillier.Add(res, c1, c2);

//     str3 = mpz_get_str(nullptr,10,res);

//     paillier.Decrypt(mm, res);

//     cout << "400000000+500000000 =" << mm << endl;

//     mpz_set_ui(res, 0);
//     mpz_set_ui(mm, 0);
//     mpz_set_ui(x, 300000000);
    
//     str4 = mpz_get_str(nullptr,10,x);

//     paillier.Mul(res, c1, x);

//     str5 = mpz_get_str(nullptr,10,res);
//     cout<< str1 <<endl;
//     cout<< str2 <<endl;
//     cout<< str3 <<endl;
//     cout<< str4 <<endl;
//     cout<< str5 <<endl;
//     paillier.Decrypt(mm, res);
//     cout << "400000000*300000000 =" << mm << endl;

//     return 0;
// }
