#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_hidden_vec_mul.h"
#include "../config.h"

void mat_hidden_vec_mul(input_stream<float> * __restrict in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    auto pout = aie::begin_vector_circular<VECTOR_LANES>(out);

    bool first_iteration_flag = true;
    alignas(32) aie::accum<accfloat, 8> accum;
    alignas(32) aie::vector<float, 8> hidden[H_VECTOR_SIZE/VECTOR_LANES];

    for (;;){
        if (first_iteration_flag) {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                hidden[i] = aie::load_v<8>((float*)&h_init[i]);
                }
                first_iteration_flag = false;
        } else {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                hidden[i] = readincr_v<8>(in);
                }
        }

        // Compute
        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 8>());
            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES; j++)
                {
                accum = aie::mac(accum, hidden[j], aie::load_v<8>((float*)&weights[i*H_VECTOR_SIZE + VECTOR_LANES*j]));
            }

            out.acquire();
            *pout++ = accum;
            out.release();

        }
    }
}