#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aggregator_kernel.h"
#include "../config.h"

void aggregator_kernel( input_pktstream *in,
                        output_buffer<float> *out
){ 
    float z_result[H_VECTOR_SIZE];
    for (int i = 0; i < H_VECTOR_SIZE; i+=DIST_COEFF){
        readincr(in); //read header and discard
        for (int j = 0; j < DIST_COEFF; j++){
            z_result[i + j] = readincr(in);
        }
    }

    for (int i = 0; i < H_VECTOR_SIZE; i++){writeincr(out, z_result[i]);}
}