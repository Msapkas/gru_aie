#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "chsg_mat_r_mul_h.h"
#include "../config.h"

// This kernel executes the U ( Weights Matrix ) multiplication with the elementwise R with Prev_Hidden_State_Vector (A Specialized kernel is needed for this)
// The Matrix-Vec multiplication is distributed row wise (each kernel performs mac operations of a row). 

void chsg_mat_r_mul_h(input_stream<float> * r_in,
                        input_stream<float> * __restrict h_in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[VECTOR_LANES*H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> hidden[H_VECTOR_SIZE/VECTOR_LANES], v_r_xelem_h[H_VECTOR_SIZE/VECTOR_LANES], reset_gate[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;
    alignas(32) aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    // First time you start the kernel you initialize the hidden state with h_init coming from RTP
    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
        {
        hidden[i] = v_hidden[i];
    }
    chess_separator_scheduler();
    // Infinite loop 
    for (;;){
        // Read r gate from aggregator
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            reset_gate[i] = readincr_v<4>(r_in);
        }
        chess_separator_scheduler();
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            v_r_xelem_h[i] = aie::mul(reset_gate[i],hidden[i]).to_vector<float>(0);
        }
        // Compute
        alignas(32) float* r_xelem_h  = (float*)&v_r_xelem_h; // cast float vector to float pointer

        for (int dist = 0; dist < DIST_COEFF; dist++) chess_loop_count(DIST_COEFF)
            {   
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE)
                {
                acc = aie::mac( acc, v_weights[i+H_VECTOR_SIZE*dist], r_xelem_h[i]);
            }
            writeincr(out, acc);
        }
        chess_separator_scheduler(H_VECTOR_SIZE);
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            hidden[i] = readincr_v<4>(h_in);
        }
        chess_separator_scheduler();
    }
}
