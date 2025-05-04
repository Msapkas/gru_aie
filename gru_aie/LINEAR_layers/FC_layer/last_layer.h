#ifndef LAST_FC_LAYER
#define LAST_FC_LAYER

#include <adf.h>
#include "../config.h"

void last_fully_connected ( input_stream<float> * input, output_stream<float> * output,
                            const float (&layer_parameters)[output_dims_0],
                            const float (&bias));

#endif