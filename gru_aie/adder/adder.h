#ifndef ADDER_H
#define ADDER_H

#include <adf.h>
#include "config.h"

void adder (input_stream <float> * __restrict x_vector,
            input_stream <float> * __restrict h_vector,
            output_pktstream * out,
            const float (&bias)[DIST_COEFF*VECTOR_LANES],
            const int (&id));

#endif