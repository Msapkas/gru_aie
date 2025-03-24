#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "adf/io_buffer/io_buffer_types.h"
#include "mat_input_vec_mul.h"
#include "../config.h"

void mat_input_vec_mul( input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[X_VECTOR_SIZE*DIST_COEFF]

){  
    alignas(32) aie::accum<accfloat,VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> x_input[X_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_weights = (aie::vector<float, VECTOR_LANES>*) &weights;

    for (;;){
        chess_separator_scheduler();
        // Read the input and keep it
        for (int i = 0; i < X_VECTOR_SIZE/VECTOR_LANES; i++){
            x_input[i] = readincr_v<4>(in);
        }
        // chess_separator_scheduler();
        for (int i = 0; i < DIST_COEFF; i++)
        {   acc = aie::zeros<accfloat, VECTOR_LANES>();
            for (int j = 0; j < X_VECTOR_SIZE/VECTOR_LANES ; j++)
                {
                acc = aie::mac(acc,
                                x_input[j],
                                v_weights[i*(X_VECTOR_SIZE/VECTOR_LANES) + j]
                                );
            }
            writeincr(out, acc);
        }
        chess_separator_scheduler();
    }
}

