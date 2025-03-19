#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aggregator.h"
#include "../config.h"

void aggregator(  input_pktstream * in,
                  output_stream<float> * out
)

{ 
    alignas(32) float aggregated_vector[H_VECTOR_SIZE];
    alignas(32) float input;
    alignas(32) unsigned int idx;

    for (;;){
        chess_separator_scheduler();
        for (int i = 0; i < H_VECTOR_SIZE; i++){
            readincr(in); //read header and discard
            idx = readincr(in);
            // Cast - THE BITS THAT ARE INSIDE MEMORY LOCATION = &input -> TO FLOAT
            input = static_cast<float>(readincr(in));
            // unsigned int* src = (unsigned int*)& input;
            // float* dest = (float*) src;
            //
            aggregated_vector[idx] = input;
        }
        for (int i = 0; i < H_VECTOR_SIZE; i++){
            writeincr(out, aggregated_vector[i]+i*0.1);
        }
        chess_separator_scheduler();
    }
}