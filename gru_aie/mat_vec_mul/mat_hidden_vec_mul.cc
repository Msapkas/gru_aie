#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_hidden_vec_mul.h"
#include "../config.h"

void mat_hidden_vec_mul(input_stream<float> * __restrict in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    bool first_iteration_flag = true;
    alignas(32) aie::accum<accfloat, 4> accum;
    alignas(32) float hidden[H_VECTOR_SIZE];

    for (;;){
        if (first_iteration_flag) {
            for (int i = 0; i < H_VECTOR_SIZE; i++){
                hidden[i] = h_init[i];
                }
                first_iteration_flag = false;
        } else {
            for (int i = 0; i < H_VECTOR_SIZE; i++){
                hidden[i] = readincr(in);
                }
        }

        // Compute
        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 4>());
            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES; j++)
                {
                accum = aie::mac(accum, 
                                aie::load_v<4>((float*)&hidden[j]), 
                                aie::load_v<4>((float*)&weights[i*H_VECTOR_SIZE + VECTOR_LANES*j]));
            }

            aie::vector<float, 4> res = accum.to_vector<float>(0);
            float* pout = (float*)&res;
            for (int i = 0; i < VECTOR_LANES; i++){
                writeincr(out, *pout++);
            }

            // writeincr_v4(out, accum);

        }
    }
}