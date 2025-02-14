#ifndef MAT_INPUT_VEC_MUL_H
#define MAT_INPUT_VEC_MUL_H
#include "./config.h"

// template<int VECTOR_SIZE, int VECTOR_LANES, int MULT_DIST_COEFF> 
void mat_input_vec_mul(     adf::input_circular_buffer           <float,adf::extents<X_VECTOR_SIZE>>            & __restrict in,
                            adf::output_async_circular_buffer    <float,adf::extents<MULT_DIST_COEFF*VECTOR_LANES>>  & __restrict out,
                            const float (&weights)[X_VECTOR_SIZE*MULT_DIST_COEFF]);

#endif