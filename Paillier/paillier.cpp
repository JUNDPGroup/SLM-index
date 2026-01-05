#include "paillier.h"
#include <gmpxx.h>
#include <stdexcept>
#include <math.h>
extern gmp_randstate_t gmp_rand;
gmp_randstate_t gmp_rand;

// void Paillier::enc_clear()
// {
//     mpz_clears(p, q, g, n, nsquare, lambda, lmdInv, NULL);
//     std::cout << "释放内存并推出程序......" << std::endl;
// }

void Paillier::KeyGen(unsigned long bitLen)
{
    gmp_randinit_default(gmp_rand);
    mpz_t r;
    mpz_init(r);
    mpz_rrandomb(r, gmp_rand, bitLen); // r <--- rand
    mpz_nextprime(p, r);               // p是大素数
    mpz_set(r, p);
    mpz_nextprime(q, r); // q是大素数

    mpz_mul(n, p, q);       // n = p*q
    mpz_add_ui(g, n, 1);    // g = n+1
    mpz_mul(nsquare, n, n); // nsqaure = n * n;

    mpz_sub_ui(p, p, 1); // p = p-1
    mpz_sub_ui(q, q, 1); // q = q-1
    mpz_lcm(lambda, p, q);         // lambda = lcm(p-1, q-1)
    //mpz_mul(lambda, p, q);
    mpz_invert(lmdInv, lambda, n); // lmdInv = lambda^{-1} mod n
    mpz_clear(r);
}

void Paillier::Encrypt(mpz_t c, mpz_t m)
{

    if (mpz_cmp_ui(m, 0) < 0) { // 处理负数输入
        mpz_add(m, m, n);
    }

    if (mpz_cmp(m, n) >= 0)
    {

        throw("m must be less than n");
        return;
    }


    mpz_t r;
    mpz_init(r);
    gmp_randinit_default(gmp_rand);
    mpz_urandomm(r, gmp_rand, n); // r <--- rand

    mpz_powm(c, g, m, nsquare); // c = g^m mod n^2
    mpz_powm(r, r, n, nsquare); // r = r^n mod n^2
    mpz_mul(c, c, r);           // c = c*r
    mpz_mod(c, c, nsquare);     // c = c mod n^2

    mpz_clear(r);
}

void Paillier::Decrypt(mpz_t m, mpz_t c)
{
    if (mpz_cmp(c, nsquare) >= 0)
    {
        throw("ciphertext must be less than n^2");
        return;
    }
    mpz_powm(m, c, lambda, nsquare); // c = c^lambda mod n^2
    // m = (c - 1) / n * lambda^(-1) mod n

    mpz_sub_ui(m, m, 1);   // c=c-1
    mpz_fdiv_q(m, m, n);   // c=(c-1)/n
    mpz_mul(m, m, lmdInv); // c=c*lambda^(-1)
    mpz_mod(m, m, n);      // m=c mod n
}

void Paillier::Add(mpz_t res, mpz_t c1, mpz_t c2)
{

    if (mpz_cmp(c1, nsquare) >= 0)
    {
        throw("Add: ciphertext must be less than n^2");
        return;
    }
    if (mpz_cmp(c2, nsquare) >= 0)
    {
        throw("Add: ciphertext must be less than n^2");
        return;
    }
    mpz_mul(res, c1, c2);
    mpz_mod(res, res, nsquare);
}


void Paillier::Sub(mpz_t res, mpz_t c1, mpz_t c2) 
{
    // if (mpz_cmp(c1, nsquare) >= 0 || mpz_cmp(c2, nsquare) >= 0) {
    //     throw std::invalid_argument("ciphertext must be less than n^2");
    // }
    // mpz_t inv_c2;
    // mpz_init(inv_c2);
    // if (mpz_invert(inv_c2, c2, nsquare) == 0) {
    //     mpz_clear(inv_c2);
    //     throw std::runtime_error("Modular inverse does not exist");
    // }
    // mpz_mul(res, c1, inv_c2);
    // mpz_mod(res, res, nsquare);
    // mpz_clear(inv_c2);
    if (mpz_cmp(c1, nsquare) >= 0 || mpz_cmp(c2, nsquare) >= 0) {
        throw std::invalid_argument("Sub: ciphertext must be less than n^2");
    }
    mpz_t minus_one, neg_c2;
    mpz_inits(minus_one, neg_c2, nullptr);
    mpz_set_si(minus_one, 1);
    mpz_class n_1(mpz_class(n) - mpz_class(minus_one));

    Mul(neg_c2, c2, n_1.get_mpz_t()); 
    Add(res, c1, neg_c2); 
    mpz_mod(res, res, nsquare);
    mpz_clears(minus_one, neg_c2, nullptr);
}


// 只能是同态标量乘
void Paillier::Mul(mpz_t res, mpz_t c, mpz_t e)
{
    if (mpz_cmp(c, nsquare) >= 0)
    {
        throw("Mul: ciphertext must be less than n^2");
        return;
    }
    if (mpz_cmp(e, n) >= 0)
    {
        throw("exponent must be less than n");
    }
    // mpz_powm(res, c, e, nsquare);
    if (mpz_cmp_ui(e, 0) < 0) {  // 处理负指数
        mpz_invert(e, e, n);
        mpz_abs(e, e);
    }
    mpz_powm(res, c, e, nsquare);
}


// mpz_class Enc_float(float x, Paillier& paillier)
// {
//     x = std::round(x * 1000000.0f) / 1000000.0f;
//     mpz_class X(x * 1e6), enc_x;
//     paillier.Encrypt(enc_x.get_mpz_t(), X.get_mpz_t());
// }

