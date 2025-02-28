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
    alignas(32) aie::vector<float, 4> old_hidden_state[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::vector<float, 4> z_v, hhat_v;
    alignas(32) float hhat[H_VECTOR_SIZE], z[H_VECTOR_SIZE];
    alignas(32) aie::accum<accfloat, 4> new_hidden_state[H_VECTOR_SIZE/VECTOR_LANES];

    for (;;){

        if (first_iteration_flag) {
            first_iteration_flag = false;
            // Data acquisition
            for (int i = 0; i < H_VECTOR_SIZE; i++){
                z[i] = readincr(z_in);
                hhat[i] = readincr(cand_hidden_state_in);
            }
            // Compute using h_init
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                
                z_v = aie::load_v<4>((float*)&z[i]);
                hhat_v = aie::load_v<4>((float*)&hhat[i]);

                new_hidden_state[i] = aie::mul(aie::sub(float(1) ,z_v), hhat_v);

                new_hidden_state[i] = aie::mac(new_hidden_state[i], z_v, aie::load_v<4>((float*)&h_init[i+VECTOR_LANES]));
                
                old_hidden_state[i] = new_hidden_state[i].to_vector<float>(0); // Keep the new hidden state in memory for later, needs to be vector for the mac
            }

        } else {
            // Data acquisition
            for (int i = 0; i < H_VECTOR_SIZE; i++){
                z[i] = readincr(z_in);
                hhat[i] = readincr(cand_hidden_state_in);
            }
            // Compute using old_hidden_state
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){

                z_v = aie::load_v<4>((float*)&z[i]);
                hhat_v = aie::load_v<4>((float*)&hhat[i]);

                new_hidden_state[i] = aie::mul(aie::sub(float(1),z_v), hhat_v);

                new_hidden_state[i] = aie::mac(new_hidden_state[i], z_v, old_hidden_state[i]);

                old_hidden_state[i] = new_hidden_state[i].to_vector<float>(0); // Keep the new hidden state in memory for later needs to be vector for the mac
            }
        }
    }

    // output new hidden state
    float *pout = (float*) &old_hidden_state[0];
    for (int i = 0; i < H_VECTOR_SIZE; i++) {
        writeincr(new_hidden_state_out, *pout++);
        printf("nhs output: \n");
    }
}
