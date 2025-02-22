#ifndef SIGMOID_REDUCE_H
#define SIGMOID_REDUCE_H
#include <adf.h>
#include "../config.h"

void sigmoid_reduce(adf::input_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>, adf::margin<VECTOR_LANES>>& __restrict x_in,
                    adf::input_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>, adf::margin<VECTOR_LANES>>& __restrict h_in,
                    output_pktstream *out,
                    const float (&bias)[DIST_COEFF]);

#endif