#ifndef SIGMOID_REDUCE_H
#define SIGMOID_REDUCE_H
#include <adf.h>
#include "../config.h"

void sigmoid_reduce(input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF],
                    const unsigned int (&identifier));

#endif