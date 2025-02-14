#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid_reduce.h"
#include "act_funcs_luts/sigm.h"
#include "./config.h"

void sigmoid_reduce(adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict h_in,
                adf::output_async_circular_buffer <float,adf::extents<MULT_DIST_COEFF>> & __restrict out,
                const float (&bias)[MULT_DIST_COEFF]

){  auto x_pin = aie::begin_vector_circular<VECTOR_LANES>(x_in);
    auto h_pin = aie::begin_vector_circular<VECTOR_LANES>(h_in);

    float res[MULT_DIST_COEFF];
    float gate_result[MULT_DIST_COEFF]; 

    for(;;){
        for (int i = 0; i < MULT_DIST_COEFF; i++){
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES){
                res[i] = aie::reduce_add(*x_pin++) + aie::reduce_add(*h_pin++) + bias[i];
            }
        }
        
        for (int i = 0; i < MULT_DIST_COEFF; i++)chess_unroll_loop(*){
        gate_result[i] = sigm[res[i]];
        }
    }
}