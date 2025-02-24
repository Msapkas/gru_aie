#ifndef MAT_HIDDEN_VEC_MUL_H
#define MAT_HIDDEN_VEC_MUL_H
#include "../config.h"

void mat_hidden_vec_mul(input_stream<float> * __restrict in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]);

#endif