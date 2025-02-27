#ifndef R_AGGREGATOR_MUL_H_H
#define R_AGGREGATOR_MUL_H_H

#include <adf.h>
#include "../config.h"

void r_aggregator_mul_h(input_pktstream * r_in,
                        input_stream<float> * __restrict h_in,
                        output_stream<float> * __restrict r_mul_h_out,
                        const float (&h_init)[H_VECTOR_SIZE]);

#endif