#ifndef AGGREGATOR_H
#define AGGREGATOR_H

#include <adf.h>
#include "../config.h"

void aggregator(  input_pktstream * in,
                  output_stream<float> * out);

#endif