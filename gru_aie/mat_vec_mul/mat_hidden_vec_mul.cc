#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_hidden_vec_mul.h"
#include "../config.h"

// This kernel executes the U h ( Weights Matrix - Hidden State Vector ) multiplication. The Matrix-Vec multiplication is distributed row wise (each kernel performs mac operations of a row). 

void mat_hidden_vec_mul(input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[DIST_COEFF*H_VECTOR_SIZE*VECTOR_LANES],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) float hidden[H_VECTOR_SIZE];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;

    for (int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE)
        {
        hidden[i] = h_init[i];
    }
    chess_separator_scheduler(H_VECTOR_SIZE);
    for (;;){
        // Compute
        chess_separator_scheduler();
        for (int dist = 0; dist < DIST_COEFF; dist++) chess_loop_count(DIST_COEFF)
            {
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE) chess_prepare_for_pipelining 
                {
                acc = aie::mac( acc, v_weights[i+H_VECTOR_SIZE*dist], hidden[i]);
            }
            writeincr(out, acc);
        }
        chess_separator_scheduler(VECTOR_LANES);
        for (int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE)
            {
            hidden[i] = readincr(in);
        }
    }
}
