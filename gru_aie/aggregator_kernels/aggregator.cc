#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aggregator.h"
#include "../config.h"

void aggregator(  input_pktstream * in, output_stream<float> * out )
{ 
    alignas(32) float aggregated_vector[H_VECTOR_SIZE];
    int dummy;     
  
    for (;;){
        chess_separator_scheduler();
        for (int i = 0; i < H_VECTOR_SIZE; i++)
        // insert loop count
        {
            dummy = readincr(in); //read header and discard
            int idx = int(readincr(in));
            // Cast - THE BITS THAT ARE INSIDE MEMORY LOCATION = &input -> TO FLOAT
            unsigned int input = readincr(in);
            unsigned int* src = (unsigned int*)& input;
            float* dest = (float*) src;
            //
            aggregated_vector[idx] = *dest;
        }
        
        for (int i = 0; i < H_VECTOR_SIZE; i++) 
        {
            writeincr(out, aggregated_vector[i]);
        }
        chess_separator_scheduler();
    }
}