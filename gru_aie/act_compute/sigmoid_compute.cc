#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid_compute.h"
#include "../config.h"

void sigmoid_compute(input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF],
                    const int (&identifier)

){  
    float partial_result[DIST_COEFF], input_Wx[DIST_COEFF], input_Uh[DIST_COEFF], final_result[DIST_COEFF];
    aie::vector<float, 4> poly_vector;
    alignas(32) const float constants[4] = {1.0 , 12.0/32.0, -1.0/32.0, 0.0};
    aie::vector<float, 4> constants_vector = aie::load_v<4>(constants);

    // Some header values
    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); // for output pktstream the ID of a merge is 0

    // infinite loop
    for (;;){
        chess_separator_scheduler();
        for (int i = 0; i < DIST_COEFF; i++){
            input_Wx[i] = readincr(x_in);
            input_Uh[i] = readincr(h_in);
        }
        chess_separator_scheduler(6);
        for (int i = 0; i < DIST_COEFF; i++){
            partial_result[i] = bias[i] + input_Wx[i] + input_Uh[i];
            // Compute act func
            poly_vector.set(0.5, 0);

            poly_vector.set(partial_result[i], 1);

            float res_2 = partial_result[i]*partial_result[i];

            poly_vector.set(res_2*partial_result[i], 2);
            // vector_mem_load[3] = vector_mem_load[2]*res_2;
            poly_vector.set(0.0, 3);

            aie::accum<accfloat, 4> poly = aie::mul(poly_vector, constants_vector);
            final_result[i] = aie::reduce_add(poly.to_vector<float>(0));
        }
        for (int i = 0; i < DIST_COEFF; i++){
            writeHeader(out,pktType,ID);
            writeincr(out, identifier + i);
            writeincr(out,final_result[i],true);
        }
    }
}
