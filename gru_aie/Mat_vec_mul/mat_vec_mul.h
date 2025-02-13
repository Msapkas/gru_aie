#ifndef mat_vec_mul_h
#define mat_vec_mul_h
#include "./config.h"

void mat_vec_mul(   adf::input_circular_buffer           <float,adf::extents<X_SIZE>>      & __restrict in,
                    adf::output_async_circular_buffer    <float,adf::extents<DIST_COEFF*VECTOR_SIZE>> & __restrict out,
                    const float (&weights)[X_SIZE*DIST_COEFF]);

#endif