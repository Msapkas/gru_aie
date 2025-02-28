#ifndef NEW_HIDDEN_STATE_H
#define NEW_HIDDEN_STATE_H
#include "../config.h"

void new_hidden_state(  input_stream<float> * __restrict cand_hid_state_in,
                        input_stream<float> * __restrict z_in,
                        output_stream<float> * __restrict new_hidden_state_out,
                        const float (&h_init)[H_VECTOR_SIZE]);

#endif