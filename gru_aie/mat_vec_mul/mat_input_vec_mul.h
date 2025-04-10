#ifndef MAT_INPUT_VEC_MUL_H
#define MAT_INPUT_VEC_MUL_H
#include "../config.h"

void mat_input_vec_mul( input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[VECTOR_LANES*X_VECTOR_SIZE*DIST_COEFF]);

#endif