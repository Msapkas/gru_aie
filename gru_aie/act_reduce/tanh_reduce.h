#ifndef TANH_REDUCE_H
#define TANH_REDUCE_H
#include "config.h"

void tanh_reduce(adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict h_in,
                adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF>> & __restrict out,
                const float (&bias)[DIST_COEFF]);

#endif