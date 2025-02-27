#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "chsg_mat_r_mul_h.h"
#include "../config.h"

void chsg_mat_r_mul_h(input_stream<float> * __restrict r_in,
                        input_stream<float> * __restrict h_in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict out,
                        const float (&weights)[H_VECTOR_SIZE*DIST_COEFF],
                        const float (&h_init)[H_VECTOR_SIZE]

){  
    auto pout = aie::begin_vector_circular<VECTOR_LANES>(out);

    bool first_iteration_flag = true;
    alignas(32) aie::accum<accfloat, 8> accum;
    alignas(32) aie::vector<float, 8> hidden[H_VECTOR_SIZE/VECTOR_LANES];
    alignas(32) aie::vector<float, 8> r_mul_h;
    alignas(32) float reset_gate[H_VECTOR_SIZE];

    for (;;){
        if (first_iteration_flag) {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                hidden[i] = aie::load_v<8>((float*)&h_init[i]);
                }
                first_iteration_flag = false;
        } else {
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
                hidden[i] = readincr_v<8>(h_in);
                }
        }

        for (int i = 0; i < H_VECTOR_SIZE; i++){
            reset_gate[i] = readincr(r_in);
        }

        // Compute
        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 8>());
            for (int j = 0; j < H_VECTOR_SIZE/VECTOR_LANES; j++)
                {
                r_mul_h = aie::mul(aie::load_v<8>((float*)&reset_gate[j]), hidden[j]).to_vector<float>(0);
                accum = aie::mac(accum, r_mul_h, aie::load_v<8>((float*)&weights[i*H_VECTOR_SIZE + VECTOR_LANES*j]));
            }

            out.acquire();
            *pout++ = accum;
            out.release();

        }
    }
}