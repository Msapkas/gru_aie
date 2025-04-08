#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "master_aggregator.h"
#include "../config.h"

void master_aggregator( input_stream<float> * __restrict in_0,
                        input_stream<float> * __restrict in_1,
                        output_stream<float> * __restrict out)
{   
    alignas(32) aie::vector<float, VECTOR_LANES> final_aggregated_vector[H_VECTOR_SIZE/VECTOR_LANES]; 
    // constexpr int K_DIV = H_VECTOR_SIZE/2

    for (;;){
        for (int i = 0; i< H_VECTOR_SIZE/VECTOR_LANES/2; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES) {
            final_aggregated_vector[i] = readincr_v<4, aie_stream_resource_in::a>(in_0);
            final_aggregated_vector[i+NKERNELS/2] = readincr_v<4, aie_stream_resource_in::a>(in_1);
        }
        chess_separator_scheduler();
        for (int i = 0; i< H_VECTOR_SIZE/VECTOR_LANES; i++) chess_loop_count(H_VECTOR_SIZE/VECTOR_LANES) {
            writeincr(out, final_aggregated_vector[i]);
        }
        chess_separator_scheduler(H_VECTOR_SIZE);
    }

}