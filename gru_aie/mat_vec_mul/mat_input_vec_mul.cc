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
    alignas(32) aie::accum<accfloat, 4> accum;
    alignas(32) float x_input[X_VECTOR_SIZE];

    for (;;){

        // Read the input and keep it
        for (int i = 0; i < X_VECTOR_SIZE; i++){
            x_input[i] = readincr(in);
        }

        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 4>());
            for (int j = 0; j < X_VECTOR_SIZE/VECTOR_LANES; j++)
                {
                accum = aie::mac(accum,
                                aie::load_v<4>((float*)&x_input[i]),
                                aie::load_v<4>((float*)&weights[i*X_VECTOR_SIZE + VECTOR_LANES*j]));
            }

            aie::vector<float, 4> res = accum.to_vector<float>(0);
            float* pout = (float*)&res;
            for (int i = 0; i < VECTOR_LANES; i++){
                writeincr(out, *pout++);
            }
        }
    }
}
