//
//  Description:		source for computing Z-order curve value
//  Created:			2016/12/11
//  Last updated:		2019/10/17
//  Author:				Yanchuan Chang
//  Mail:				changyanchuan@gmail.com
//

#ifndef Z_H
#define Z_H

#include<stdlib.h>

// __uint128_t compute_Z_value(long long x[], const size_t& x_len, const long long& bits);
__uint128_t compute_Z_value(const long long* coords, int dim, int bits_per_dim);
long long compute_Z_value(long long x, long long y, int bit_num);
#endif
