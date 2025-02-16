#ifndef AGGREGATOR_KERNEL_H
#define AGGREGATOR_KERNEL_H

#include <adf.h>
#include "../config.h"

void aggregator_kernel( input_pktstream *in,
                        output_stream<float> *out);


#endif