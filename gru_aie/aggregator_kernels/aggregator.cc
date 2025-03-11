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
  
    for (;;){
        chess_separator_scheduler();
        for (int i = 0; i < H_VECTOR_SIZE; i++){
            // printf("read 1 bef, %i \n", i);
            readincr(in); //read header and discard
            // printf("read 1 after, %i \n", i);
            unsigned int idx = readincr(in);
            // printf("read 2 after, %i \n", i);
            // Cast - THE BITS THAT ARE INSIDE MEMORY LOCATION = &input -> TO FLOAT
            //jesus christ
            unsigned int input = readincr(in);
            //  printf("read 3 after, %i \n", i);
            unsigned int* src = (unsigned int*)& input;
            float* dest = (float*) src;
            //
            aggregated_vector[idx] = *dest;
            //  printf("end loop, %i \n", i);
        }
        
        // printf("check 1, %i \n", k);
        for (int k = 0; k < H_VECTOR_SIZE; k++){
            // printf("out bef, %i \n", i);
            writeincr(out, aggregated_vector[k]+0.1*k);
            // printf("out aft, %i \n", i);
        }
        chess_separator_scheduler();
    }
}