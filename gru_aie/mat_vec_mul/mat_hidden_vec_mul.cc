#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_hidden_vec_mul.h"
#include "../config.h"

void mat_hidden_vec_mul(input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    bool first_iteration_flag = true;
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> hidden[H_VECTOR_SIZE/VECTOR_LANES];

    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;
    alignas(32) aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    for (;;){
        chess_separator_scheduler();
        if (first_iteration_flag) {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++)chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
                {
                hidden[i] = v_hidden[i];
            }
                first_iteration_flag = false;
        } else {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++)chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
                {
                hidden[i] = readincr_v<4>(in);
            }
        }
        chess_separator_scheduler();
        // Compute
        for (int i = 0; i < DIST_COEFF; i++)chess_loop_count(DIST_COEFF)
            {
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES ; j++)chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
                {
                acc = aie::mac( acc, 
                                hidden[j],
                                v_weights[i*(H_VECTOR_SIZE/VECTOR_LANES) + j]
                                );
            }
            writeincr(out, acc);
        }
        chess_separator_scheduler();
    }
}
