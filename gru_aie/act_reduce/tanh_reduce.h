#ifndef TANH_REDUCE_H
#define TANH_REDUCE_H
#include "config.h"

void tanh_reduce(adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict h_in,
                adf::output_pktstream *out,
                const float (&bias)[MULT_DIST_COEFF]);

#endif