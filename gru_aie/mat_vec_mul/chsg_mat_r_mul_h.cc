#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "chsg_mat_r_mul_h.h"
#include "../config.h"

void chsg_mat_r_mul_h(input_stream<float> * __restrict r_in,
                        input_stream<float> * __restrict h_in,
                        output_stream<float> * __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    bool first_iteration_flag = true;
    alignas(32) aie::accum<accfloat, 4> accum;
    alignas(32) aie::vector<float, 4> r_mul_h;
    alignas(32) float hidden[H_VECTOR_SIZE], reset_gate[H_VECTOR_SIZE];

    for (;;){
        if (first_iteration_flag) {
            for (int i = 0; i < H_VECTOR_SIZE; i++){
                hidden[i] = h_init[i];
                }
                first_iteration_flag = false;
        } else {
            for (int i = 0; i < H_VECTOR_SIZE; i++){
                hidden[i] = readincr(h_in);
                }
        }

        for (int i = 0; i < H_VECTOR_SIZE; i++){
            reset_gate[i] = readincr(r_in);
        }

        // Compute
        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 4>());

            for (int j = 0; j < (H_VECTOR_SIZE - 1); j += (VECTOR_LANES - 1))
                {
                r_mul_h = aie::mul(aie::load_v<4>((float*)&reset_gate[j]), hidden[j]).to_vector<float>(0);

                accum = aie::mac(accum, r_mul_h, aie::load_v<4>((float*)&weights[i*(H_VECTOR_SIZE - 1) + j]));
            }

            aie::vector<float, 4> res = accum.to_vector<float>(0);
            float* pout = (float*)&res;
            for (int i = 0; i < VECTOR_LANES; i++){
                writeincr(out, *pout++);
            }

        }
    }
}