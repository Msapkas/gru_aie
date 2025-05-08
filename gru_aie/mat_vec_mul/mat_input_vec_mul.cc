#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "adf/io_buffer/io_buffer_types.h"
#include "mat_input_vec_mul.h"
#include "../config.h"

// This kernel executes the W x ( Weights Matrix - Input Vector ) multiplication. The Matrix-Vec multiplication is distributed row wise (each kernel performs mac operations of a row). 

void mat_input_vec_mul( input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[X_VECTOR_SIZE*DIST_COEFF]

){  
    aie::accum<accfloat,VECTOR_LANES> acc;
    aie::vector<float, VECTOR_LANES> x_input[X_VECTOR_SIZE/VECTOR_LANES];
    aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;

    for (;;){
        chess_separator_scheduler(); // Separators are crucial for the correct scheduling of the kernel
        // Read the input
        for (int i = 0; i < X_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(X_VECTOR_SIZE/VECTOR_LANES)
            {
            x_input[i] = readincr_v<4>(in);
        }
        chess_separator_scheduler(X_VECTOR_SIZE);
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF) // For each row
            {   
            acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int j = 0; j < X_VECTOR_SIZE/VECTOR_LANES ; j++) chess_loop_count(X_VECTOR_SIZE/VECTOR_LANES)
                {
                acc = aie::mac(acc,
                                x_input[j],
                                v_weights[i*(X_VECTOR_SIZE/VECTOR_LANES) + j]
                                );
            }
            writeincr(out, aie::reduce_add( acc.to_vector<float>(0)) ); // Write the output, which is a VECTOR LANE length vector
        }
        chess_separator_scheduler(VECTOR_LANES); 
    }
}

