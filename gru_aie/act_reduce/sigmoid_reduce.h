#ifndef SIGMOID_REDUCE_H
#define SIGMOID_REDUCE_H
#include "config.h"

void sigmoid_reduce(adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict h_in,
                adf::output_pktstream *out,
                const float (&bias)[DIST_COEFF]);

#endif