#ifndef CHSG_MAT_R_MUL_H_H
#define CHSG_MAT_R_MUL_H_H
#include "../config.h"

void chsg_mat_r_mul_h(input_stream<float> * __restrict r_in,
                        input_stream<float> * __restrict h_in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]);

#endif