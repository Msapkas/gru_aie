#ifndef MASTER_AGGREGATOR_H
#define MASTER_AGGREGATOR_H

#include <adf.h>
#include "../config.h"

void master_aggregator( input_stream<float> * __restrict in_0,
                        input_stream<float> * __restrict in_1,
                        output_stream<float> * __restrict out);

#endif