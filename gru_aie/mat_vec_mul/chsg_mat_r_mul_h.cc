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
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> hidden[H_VECTOR_SIZE/VECTOR_LANES], r_xelem_h[H_VECTOR_SIZE/VECTOR_LANES], reset_gate[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;
    alignas(32) aie::vector<float, VECTOR_LANES> * v_hidden  = (aie::vector<float, VECTOR_LANES>*) &h_init;

    // First time you start the kernel you initialize the hidden state with h_init coming from RTP
    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
        {
        hidden[i] = v_hidden[i];
    }

    // Infinite loop 
    for (;;){
        // chess_separator_scheduler(); // Scheduling pragmas are important. Seperate inputs from outputs so that they do not get scheduled in the same instruction (dealocks)
        // Read r gate from aggregator
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            reset_gate[i] = readincr_v<4>(r_in);
        }
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            r_xelem_h[i] = aie::mul(reset_gate[i],hidden[i]).to_vector<float>(0);
        }
        // chess_separator_scheduler();
        // Compute
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
            {   
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES; j ++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
                {
                acc = aie::mac( acc, 
                                r_xelem_h[j],
                                v_weights[i*(H_VECTOR_SIZE/VECTOR_LANES) + j]
                                );
            }
            writeincr(out, acc);
        }
        chess_separator_scheduler();
        for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES)
            {
            hidden[i] = readincr_v<4>(h_in);
        }
        chess_separator_scheduler();
    }
}
