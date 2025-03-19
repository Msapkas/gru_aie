#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "chsg_mat_r_mul_h.h"
#include "../config.h"

void chsg_mat_r_mul_h(input_stream<float> *  r_in,
                        input_stream<float> * __restrict h_in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    bool first_iteration_flag = true;
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> hidden[H_VECTOR_SIZE/VECTOR_LANES], r_xelem_h[H_VECTOR_SIZE/VECTOR_LANES];
    float reset_gate[H_VECTOR_SIZE];

    aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;
    aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    for (;;){
        chess_separator_scheduler();
        if (first_iteration_flag) {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                hidden[i] = v_hidden[i];
                }
                first_iteration_flag = false;
        } else {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                hidden[i] = readincr_v<4>(h_in);
                }
        }     
        for (int i = 0; i < H_VECTOR_SIZE; i++){
            reset_gate[i] = readincr(r_in);
        }
        aie::vector<float, VECTOR_LANES> * v_reset_gate  = (aie::vector<float, VECTOR_LANES>*) &reset_gate;
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
            r_xelem_h[i] = aie::mul(v_reset_gate[i],hidden[i]).to_vector<float>(0);
        }
        // Compute
        for (int i = 0; i < DIST_COEFF; i++)
        {   acc = aie::zeros<accfloat, VECTOR_LANES>();

            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES; j ++){
                acc = aie::mac( acc, 
                                r_xelem_h[j],
                                v_weights[i*(H_VECTOR_SIZE/VECTOR_LANES) + j]
                                );
            }

            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                writeincr(out, acc);
            }
        }
        chess_separator_scheduler();
    }
}