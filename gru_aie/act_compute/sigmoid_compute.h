#ifndef SIGMOID_COMPUTE_H
#define SIGMOID_COMPUTE_H
#include <adf.h>
#include "../config.h"

void sigmoid_compute(input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF],
                    const int (&identifier));
#endif