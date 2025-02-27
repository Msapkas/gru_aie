#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aggregator.h"
#include "../config.h"

void aggregator(  input_pktstream * in,
                    output_stream<float> * __restrict out
){ 
    alignas(32) float aggregated_vector[H_VECTOR_SIZE];
    for (int i = 0; i < NKERNELS; i++){
        for (int j = 0; j < DIST_COEFF; j++){
            readincr(in); //read header and discard
            aggregated_vector[i*DIST_COEFF + j] = readincr(in);
        }
    }

    for (int i = 0; i < H_VECTOR_SIZE; i++){
        writeincr(out, aggregated_vector[i]);
    }
}