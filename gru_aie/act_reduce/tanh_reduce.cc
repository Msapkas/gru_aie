#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "tanh_reduce.h"
#include "act_funcs_luts/tanh.h"
#include "./config.h"

void tanh_reduce(adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict h_in,
                adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF>> & __restrict out,
                const float (&bias)[DIST_COEFF]

){  auto x_pin = aie::begin_vector_circular<VECTOR_LANES>(x_in);
    auto h_pin = aie::begin_vector_circular<VECTOR_LANES>(h_in);

    float res[DIST_COEFF];
    float gate_result[DIST_COEFF];

    for(;;){
        for (int i = 0; i < DIST_COEFF; i++){
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES){
                res[i] = aie::reduce_add(*x_pin++) + aie::reduce_add(*h_pin++) + bias[i];
            }
        }
        
        for (int i = 0; i < DIST_COEFF; i++)chess_unroll_loop(*){
            if (res[i] < -TANH_THR){
                gate_result[i] = -1.;
            } else if (res[i] > TANH_THR){
                gate_result[i] = 1.;
            } else {
                gate_result[i] = tanh[res[i]];
            }
        }

        // Acquire lock and output
        out.acquire();
        for (int i = 0; i < DIST_COEFF; i++){
            *pout++ = gate_result[i];
        }
        out.release();
    }
}