#ifndef ADD_TANH_H
#define ADD_TANH_H
#include <adf.h>
#include "config.h"

void adder_tanh(   input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF*VECTOR_LANES],
                    const int (&identifier));

#endif