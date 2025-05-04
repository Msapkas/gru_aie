#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_hidden_vec_mul.h"
#include "../config.h"

// This kernel executes the U h ( Weights Matrix - Hidden State Vector ) multiplication. The Matrix-Vec multiplication is distributed row wise (each kernel performs mac operations of a row). 

void mat_hidden_vec_mul(input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> hidden[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;
    alignas(32) aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
        {
        hidden[i] = v_hidden[i];
    }
    chess_separator_scheduler(H_VECTOR_SIZE);
    for (;;){
        // Compute
        chess_separator_scheduler();
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
            {
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES ; j++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
                {
                acc = aie::mac( acc, 
                                hidden[j],
                                v_weights[i*(H_VECTOR_SIZE/VECTOR_LANES) + j]
                                );
            }
            writeincr(out, acc.to_vector<float>(0));
        }
        chess_separator_scheduler(VECTOR_LANES);
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            hidden[i] = readincr_v<4>(in);
        }
        chess_separator_scheduler(H_VECTOR_SIZE);
    }
}
