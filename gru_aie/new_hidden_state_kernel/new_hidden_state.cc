#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "new_hidden_state.h"
#include "../config.h"

void new_hidden_state(  input_stream<float> * __restrict cand_hidden_state_in,
                        input_stream<float> * __restrict z_in,
                        output_stream<float> * __restrict new_hidden_state_out,
                        const float (&h_init)[H_VECTOR_SIZE]
){
    bool first_iteration_flag = true;
    alignas(32) aie::vector<float, 4> old_hidden_state[H_VECTOR_SIZE/VECTOR_LANES], hhat[H_VECTOR_SIZE/VECTOR_LANES], z[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::accum<accfloat, 4> new_hidden_state[H_VECTOR_SIZE/VECTOR_LANES];

    aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    for (;;){
        chess_separator_scheduler();
        // Data acquisition
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
            z[i] = readincr_v<4>(z_in);
            hhat[i] = readincr_v<4>(cand_hidden_state_in);
        }
        if (first_iteration_flag) {
            // Compute using h_init
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) 
            // enter pragma loop unroll
                {
                new_hidden_state[i] = aie::mul(aie::sub(float(1) , z[i]), v_hidden[i]);
                new_hidden_state[i] = aie::mac(new_hidden_state[i], z[i], hhat[i]);
                old_hidden_state[i] = new_hidden_state[i].to_vector<float>(0); // Keep the new hidden state in memory for later, needs to be vector for the mac
            }
            first_iteration_flag = false;
        } else {
            // Compute using old_hidden_state
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                new_hidden_state[i] = aie::mul(aie::sub(float(1) , z[i]), old_hidden_state[i]);
                new_hidden_state[i] = aie::mac(new_hidden_state[i], z[i], hhat[i]);
                old_hidden_state[i] = new_hidden_state[i].to_vector<float>(0); // Keep the new hidden state in memory for later needs to be vector for the mac
            }
        }
        // output new hidden state
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) {
            writeincr(new_hidden_state_out, old_hidden_state[i]);
        }
        chess_separator_scheduler();
    }
}
