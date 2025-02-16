#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "new_hidden_state.h"
#include "config.h"

void new_hidden_state(  adf::input_circular_buffer        <float,adf::extents<DIST_COEFF>> & __restrict cand_hid_state_in,
                        adf::input_circular_buffer        <float,adf::extents<DIST_COEFF>> & __restrict z_in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF>> & __restrict hidden_out,
                        const float (&h_init)[DIST_COEFF]
){
    auto p_z_in = aie::begin_circular(z_in);
    auto p_cand_hid_state_in = aie::begin_circular(cand_hid_state_in);
    auto p_h_out = aie::begin_circular(hidden_out);

    bool first_iteration_flag = true;
    float hidden_state[DIST_COEFF];
    float z, h_hat;

    for (;;){
        for (int i = 0; i < DIST_COEFF; i++) {

            if (first_iteration_flag) {
                first_iteration_flag = false;
                hidden_state[i] = (1. - z)*hidden_state[i] + z*h_hat;

            } else {
                z = *p_z_in++;
                h_hat = *p_cand_hid_state_in++;
                hidden_state[i] = (1. - z)*hidden_state[i] + z*h_hat;
            }
        }

        // Acquire lock and output
        hidden_out.acquire();
        for (int i = 0; i < DIST_COEFF; i++) {
            *p_h_out++ = hidden_state[i];
        }
        hidden_out.release();
    }
}
