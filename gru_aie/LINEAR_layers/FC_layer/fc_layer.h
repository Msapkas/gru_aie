#ifndef FC_LAYER
#define FC_LAYER

#include <adf.h>
#include "../config.h"

void fully_connected ( input_stream<float> * input, output_stream<float> * output,
                        const float (&layer_parameters)[H_VECTOR_SIZE*output_dims_0],
                        const float (&bias)[output_dims_0]
                        // const int (&sequence_length)
                        );

#endif