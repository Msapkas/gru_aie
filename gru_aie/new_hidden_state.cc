#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "new_hidden_state.h"
#include "./config.h"

void new_hidden_state(  adf::input_circular_buffer        <float,adf::extents<MULT_DIST_COEFF>> & __restrict cand_hid_state_in,
                        adf::input_circular_buffer        <float,adf::extents<MULT_DIST_COEFF>> & __restrict z_in,
                        adf::output_async_circular_buffer <float,adf::extents<MULT_DIST_COEFF>> & __restrict hidden_out,
                        const float (&h_init)[MULT_DIST_COEFF]
){
    auto p_z_in = aie::begin_vector_circular<1>(z_in);
    auto p_cand_hid_state_in = aie::begin_vector_circular<1>(cand_hid_state_in);
    auto p_h_out = aie::begin_vector_circular<1>(hidden_out);

    bool first_iteration_flag true;
    float hidden_state[MULT_DIST_COEFF];

    for (;;){
        if (first_iteration_flag) {
            hidden_state = h_init;
            first_iteration_flag = false;
        }
        
        for (int i = 0; i < MULT_DIST_COEFF; i++) {
            float z =  *p_z_in++;
            float h_hat = *p_cand_hid_state_in++;
            hidden_state[i] = (1 - z)*hidden_state[i] + z*h_hat;
        }

        // Acquire lock and output
        out.acquire();
        *p_h_out++ = hidden_state[i];
        out.release();
    }
}
