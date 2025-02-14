#ifndef ADD_REDUCE_H
#define ADD_REDUCE_H
#include "config.h"

void add_reduce(adf::input_circular_buffer<float,adf::extents<X_VECTOR_SIZE>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<X_VECTOR_SIZE>>& __restrict h_in,
                adf::package_stream<float> & __restrict out,
                const float (&bias_vec)[H_VECTOR_SIZE]);

#endif