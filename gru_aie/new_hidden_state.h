#ifndef NEW_HIDDEN_STATE_H
#define NEW_HIDDEN_STATE_H
#include "./config.h"

void new_hidden_state(  adf::input_circular_buffer        <float,adf::extents<DIST_COEFF>> & __restrict cand_hid_state_in,
                        adf::input_circular_buffer        <float,adf::extents<DIST_COEFF>> & __restrict z_in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF>> & __restrict hidden_out,
                        const float (&h_init)[DIST_COEFF]);

#endif