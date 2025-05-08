#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "last_layer.h"
#include "act_reduce/act_funcs_luts/sigm.h"
#include "../config.h"

void last_fully_connected ( input_stream<float> * input, output_stream<float> * output,
                            const float (&layer_parameters)[output_dims_0],
                            const float (&bias)
){
    alignas(32) aie::accum<accfloat, VECTOR_LANES> acc;
    alignas(32) aie::vector<float, VECTOR_LANES> input_vector[output_dims_0/VECTOR_LANES];
    alignas(32) aie::vector<float, VECTOR_LANES> * v_layer_parameters = (aie::vector<float, VECTOR_LANES>*) &layer_parameters;
    alignas(32) float res;

  
    // This value pre-calculates the division needed to scale the input value between 0 - 4095, which are the indexes of the LUT
    // by precalculating this number we skip cycle consuming division and do a multiplication instead!
    static constexpr float lut_max_idx = lut_size - 1.0;
    static constexpr float sigm_m_coeff = lut_max_idx / (2 * sigm_thresh);

    for (;;){
        chess_separator_scheduler();
        // Read input from previous layer (assuming it can be divided by VECTOR_LANES)
        for ( int i = 0; i < output_dims_0/VECTOR_LANES; i++) chess_loop_count(output_dims_0/VECTOR_LANES)
            {
                input_vector[i] = readincr_v<4>(input);
        }

        // This is a very special occasion where the output is 1 neuron, so I need to MAC a single row.
        // Perform the MACs
        acc = aie::zeros<accfloat, VECTOR_LANES>();
        for (int i = 0; i < output_dims_0/VECTOR_LANES; i++) chess_loop_count(output_dims_0/VECTOR_LANES)
            {
            // aie::print(v_layer_parameters[i], true, "MAC weights:  ");
            // aie::print(input_vector[i], true, "with vector:  ");
            acc = aie::mac(acc, input_vector[i], v_layer_parameters[i]);
            // aie::print(acc, true, "Result:  ");
        }

        // I have the MACed row, in a vector of 4s. Just like the mat_input_vec_mul kernel. I need to reduce and add the bias.
        res = aie::reduce_add(acc.to_vector<float>(0)) + bias;

        // printf("Reduced Result:   %f \n", res);

        chess_separator_scheduler();
        // once calculated the result check if you fall between the threshold sigm_thresh
        if (res <= - sigm_thresh){
            writeincr(output, 0);
        } else if (res >= sigm_thresh){
            writeincr(output, 1);
        } else {
            // if you do: employ the LUT
            int index = int((res + sigm_thresh)*sigm_m_coeff) ;
            writeincr(output,sigm[index]);
            // printf("Final Result:   %f \n", sigm[index]);
        }
        chess_separator_scheduler();

    }
}