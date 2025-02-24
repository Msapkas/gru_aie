#ifndef MAT_INPUT_VEC_MUL_H
#define MAT_INPUT_VEC_MUL_H
#include "../config.h"

void mat_input_vec_mul( input_stream<float> * __restrict in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict out,
                        const float (&weights)[X_VECTOR_SIZE*DIST_COEFF]);

#endif