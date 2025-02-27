#ifndef TANH_REDUCE_H
#define TANH_REDUCE_H
#include "config.h"

void tanh_reduce(   adf::input_async_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict x_in,
                    adf::input_async_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF]);

#endif