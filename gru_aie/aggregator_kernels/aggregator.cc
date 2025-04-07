#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aggregator.h"
#include "../config.h"

// This is the aggregator kernel. Its job is to read the outputs of the sigmoids which come at a random order
// thus needing an "ID" which serves as an index. Once the aggregator has read all the packages, it will 
// broadcast the R vector to the candidate hidden state subgraph (targeting the kernel that performs the 
// hidden state element wise multiplication with R and then Matrix - Vector multiply).

// The aggreagator COULD BE OMITTED, and broadcast directly the packets to all the candidate hidden state kernels
// but this way is more structured and also it is not clear how the system will perform with highier complexity.

void aggregator(  input_pktstream * in, output_stream<float> * out )
{ 
    alignas(32) float aggregated_vector[H_VECTOR_SIZE];
    int dummy;

    for (;;){
        for (int i = 0; i < H_VECTOR_SIZE; i++)
        {
            dummy = readincr(in); // read header and discard
            int idx = int(readincr(in)); // the index is crucial that gets casted to int (unsigned int doesn't work!)
            // Cast - THE BITS THAT ARE INSIDE MEMORY LOCATION = &input -> TO FLOAT
            unsigned int input = readincr(in);
            unsigned int* src = (unsigned int*)& input;
            float* dest = (float*) src;
            //
            aggregated_vector[idx] = *dest;
        }
        chess_separator_scheduler(); // important to have a separator inbetween
        for (int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE)
        {
            writeincr(out, aggregated_vector[i]);
        }
        chess_separator_scheduler(H_VECTOR_SIZE);
    }
}
