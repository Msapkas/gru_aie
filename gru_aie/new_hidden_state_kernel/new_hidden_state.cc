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
    alignas(32) aie::vector<float, VECTOR_LANES> prev_hidden_state[H_VECTOR_SIZE/VECTOR_LANES], hhat[H_VECTOR_SIZE/VECTOR_LANES], z[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::accum<accfloat, VECTOR_LANES> new_hidden_state[H_VECTOR_SIZE/VECTOR_LANES];

    aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    // First iteration outside the infinite loop must use h_init, passed by an RTP
    // Read Z and Cand Hidden State Gate from the aggregators
    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
        { // readincr_v<4> will make sure we are reading float vectors of 4
        z[i] = readincr_v<4>(z_in);
        hhat[i] = readincr_v<4>(cand_hidden_state_in);
    }

    // Now perform (1-z)*prev_h + z*H_hat
    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
    // enter pragma loop unroll
        {
        new_hidden_state[i] = aie::mul(aie::sub(float(1) , z[i]), v_hidden[i]);
        new_hidden_state[i] = aie::mac(new_hidden_state[i], z[i], hhat[i]);
        prev_hidden_state[i] = new_hidden_state[i].to_vector<float>(0); 
        // Keep the new hidden state in memory for later, needs to be vector for the mac
    }
    chess_separator_scheduler();
    // output new hidden state
    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
        {
        writeincr(new_hidden_state_out, prev_hidden_state[i]);
        // The var is named prev_hidden_state but at this point is just the new_hidden_state in a vector form
    }
    chess_separator_scheduler(H_VECTOR_SIZE);
    // 
    for (;;){
        chess_separator_scheduler();
        // Data acquisition
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            z[i] = readincr_v<4>(z_in);
            hhat[i] = readincr_v<4>(cand_hidden_state_in);
        }
        chess_separator_scheduler(H_VECTOR_SIZE);
        // Compute using prev_hidden_state
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
                new_hidden_state[i] = aie::mul(aie::sub(float(1) , z[i]), prev_hidden_state[i]);
                new_hidden_state[i] = aie::mac(new_hidden_state[i], z[i], hhat[i]);
                prev_hidden_state[i] = new_hidden_state[i].to_vector<float>(0); // Keep the new hidden state in memory for later needs to be vector for the mac
        }
        // output new hidden state
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            writeincr(new_hidden_state_out, prev_hidden_state[i]);
        }
        chess_separator_scheduler(H_VECTOR_SIZE);
    }
}
