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
    alignas(32) aie::accum<accfloat, VECTOR_LANES> accum;
    alignas(32) aie::vector<float, VECTOR_LANES> x_input[X_VECTOR_SIZE/VECTOR_LANES];

    for (;;){

        // Read the input and keep it
        for (int i = 0; i < X_VECTOR_SIZE/VECTOR_LANES; i++){
            x_input[i] = readincr_v<4>(in);
        }

        aie::vector<float, 8> * v8_weights = (aie::vector<float, 8>*) &weights;

        for (int i = 0; i < DIST_COEFF; i++)
        {   accum = aie::zeros<accfloat, 4>();

            for (int j = 0; j < X_VECTOR_SIZE/VECTOR_LANES ; j++)
                {
                accum = aie::mac(accum,
                                x_input[j],
                                v8_weights[i]
                                );
            }

            float* pout = (float*)&accum;
            for (int i = 0; i < VECTOR_LANES; i++){
                writeincr(out, *pout++);
            }
        }
    }
}
