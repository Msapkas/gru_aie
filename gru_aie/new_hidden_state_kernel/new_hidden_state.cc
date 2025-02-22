#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "new_hidden_state.h"
#include "config.h"

void new_hidden_state(  adf::input_circular_buffer        <float,adf::extents<H_VECTOR_SIZE>> & __restrict cand_hidden_state_in,
                        adf::input_circular_buffer        <float,adf::extents<H_VECTOR_SIZE>> & __restrict z_in,
                        adf::output_async_circular_buffer <float,adf::extents<H_VECTOR_SIZE>> & __restrict new_hidden_out,
                        const float (&h_init)[H_VECTOR_SIZE]
){
    auto p_z_in = aie::begin_circular<H_VECTOR_SIZE>(z_in);
    auto p_cand_hid_state_in = aie::begin_circular<H_VECTOR_SIZE>(cand_hid_state_in);
    auto p_h_out = aie::begin_circular<H_VECTOR_SIZE>(hidden_out);

    bool first_iteration_flag = true;
    alignas(32) aie::vector<float, 8> old_hidden_state[H_VECTOR_SIZE];
    alignas(32) aie::vector<float, 8> hhat, z;
    alignas(32) aie::accum<accfloat, 8> new_hidden_state[H_VECTOR_SIZE];

    for (;;){
        if (first_iteration_flag) {
            first_iteration_flag = false;

            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                z = *p_z_in++;
                hhat = *p_cand_hid_state_in++
                new_hidden_state[i] = aie::mul(aie::sub(1,z), hhat);
                new_hidden_state[i] = aie::mac(accum, z, aie::load_v<8>((float*)&h_init[i+VECTOR_LANES]));
            }
        } else {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                z = *p_z_in++;
                hhat = *p_cand_hid_state_in++
                new_hidden_state[i] = aie::mul(aie::sub(1,z), hhat);
                new_hidden_state[i] = aie::mac(accum, z, old_hidden_state[i]);
                old_hidden_state[i] = new_hidden_state.to_vector(0);
            }
        }
    }

    // Acquire lock and output
    hidden_out.acquire();
    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) {
        *p_h_out++ = hidden_state[i];
    }
    hidden_out.release();
}
