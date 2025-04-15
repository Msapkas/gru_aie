#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "adf/io_buffer/io_buffer_types.h"
#include "mat_input_vec_mul.h"
#include "../config.h"

// This kernel executes the W x ( Weights Matrix - Input Vector ) multiplication. The Matrix-Vec multiplication is distributed column wise (each kernel performs mac operations of a column of VECTOR_LANES rows). 

void mat_input_vec_mul( input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[DIST_COEFF*X_VECTOR_SIZE*VECTOR_LANES]

){  
    alignas(32) aie::accum<accfloat,VECTOR_LANES> acc;
    alignas(32) float x_input[X_VECTOR_SIZE];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;

    for (;;){
        // Read the input
        for (int i = 0; i < X_VECTOR_SIZE; i++) chess_loop_count(X_VECTOR_SIZE)
            {
            x_input[i] = readincr(in);
        }
        chess_separator_scheduler();
        for (int dist = 0; dist < DIST_COEFF; dist++) chess_loop_count(DIST_COEFF)
            {
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int i = 0; i < X_VECTOR_SIZE; i++) chess_loop_count(X_VECTOR_SIZE)
                {
                // How you parse the weights, really depends on how you store them
                // Here, for simplicity, I assume that the weights are passed in the correct order fromt he PS
                acc = aie::mac(acc, v_weights[i+X_VECTOR_SIZE*dist], x_input[i]);
            }
            writeincr(out, acc); // Write the output, which is a VECTOR LANE length vector
        }
        chess_separator_scheduler(VECTOR_LANES);
    }
}

