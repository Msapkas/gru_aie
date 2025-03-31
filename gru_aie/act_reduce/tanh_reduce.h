#ifndef TANH_REDUCE_H
#define TANH_REDUCE_H
#include "config.h"

void tanh_reduce(   input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF],
                    const int (&identifier));

#endif