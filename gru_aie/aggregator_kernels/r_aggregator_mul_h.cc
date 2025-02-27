#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "r_aggregator_mul_h.h"
#include "../config.h"

void r_aggregator_mul_h(input_pktstream * r_in,
                        input_stream<float> * __restrict h_in,
                        output_stream<float> * __restrict r_mul_h_out,
                        const float (&h_init)[H_VECTOR_SIZE]
){
    bool first_iteration_flag = true;
    alignas(32) float r_aggregated_values[H_VECTOR_SIZE];
    alignas(32) aie::vector<float, 8> hidden[H_VECTOR_SIZE/VECTOR_LANES];

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

    for (int i = 0; i < NKERNELS; i++){
        for (int j = 0; j < DIST_COEFF; j++){
            // readincr(r_in); //read header and discard
            r_aggregated_values[i*DIST_COEFF + j] = readincr(r_in);
            }
        }

    for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES; i++){
        writeincr_v<8>(r_mul_h_out, aie::mul(aie::load_v<8>((float*)&r_aggregated_values[i]), hidden[i]).to_vector<float>(0));
        }
    }
}