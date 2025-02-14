#ifndef SIGMOID_REDUCE_H
#define SIGMOID_REDUCE_H
#include "config.h"

void sigmoid_reduce(adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict h_in,
                adf::output_async_circular_buffer <float,adf::extents<MULT_DIST_COEFF>> & __restrict *out,
                const float (&bias)[MULT_DIST_COEFF]);

#endif