#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "fc_layer.h"
#include "../config.h"

void fully_connected ( input_stream<float> * input, output_stream<float> * output,
                        const float (&layer_parameters)[H_VECTOR_SIZE*output_dims_0],
                        const float (&bias)[output_dims_0]
                        
) {
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc[output_dims_0/VECTOR_LANES];
    alignas(32) aie::vector<float, VECTOR_LANES> result[output_dims_0/VECTOR_LANES];
    alignas(32) float input_vector[H_VECTOR_SIZE];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_layer_parameters = (aie::vector<float, VECTOR_LANES>*) &layer_parameters;
    alignas(32) aie::accum<accfloat, VECTOR_LANES> * a_bias = (aie::accum<accfloat, VECTOR_LANES>*) &bias;

    for (;;){

        for (int seq = 0; seq < (sequence_length - 1); seq ++) chess_loop_count(sequence_length - 1)
            {
            for ( int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE)
                {
                readincr(input);
            }
        }

        for ( int i = 0; i < H_VECTOR_SIZE; i++) chess_loop_count(H_VECTOR_SIZE)
            {
                input_vector[i] = readincr(input);
        }

        for (int i = 0; i < output_dims_0/VECTOR_LANES; i++) chess_loop_count(output_dims_0/VECTOR_LANES) // For 4 rows, mac column wise.
            {  
            acc[i] = a_bias[i];
            for (int j = 0; j < H_VECTOR_SIZE; j++) chess_loop_count(H_VECTOR_SIZE)
                {
                acc[i] = aie::mac(acc[i], v_layer_parameters[j], input_vector[j]);
            }
        }

        // apply relu

        for (int i = 0; i < output_dims_0/VECTOR_LANES; i++) chess_loop_count(output_dims_0/VECTOR_LANES)
            {
            result[i] = aie::max(float(0), acc[i].to_vector<float>(0));
        }

        // output 
        chess_separator_scheduler();
        for (int i = 0; i < output_dims_0/VECTOR_LANES; i++) chess_loop_count(output_dims_0/VECTOR_LANES)
            {
            writeincr(output, acc[i]); // Write the output
        }
        chess_separator_scheduler(output_dims_0);
    }
}