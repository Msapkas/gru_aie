#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "adder.h"
#include "../config.h"

void adder (input_stream <float> * __restrict x_vector_in, 
            input_stream <float> * __restrict h_vector_in, 
            output_pktstream * out,
            const float (&bias)[DIST_COEFF*VECTOR_LANES],
            const int (&id)
){
    for (;;){
        alignas(128) aie::vector<float, VECTOR_LANES> x[DIST_COEFF], h[DIST_COEFF], result[DIST_COEFF];
        alignas(128) aie::vector<float, VECTOR_LANES> * v_bias = (aie::vector<float, VECTOR_LANES>*)&bias;

        static const unsigned int pktType = 0;

        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
            {
            x[i] = readincr_v<4>(x_vector_in);
            result[i] = aie::add(x[i], v_bias[i]);
            h[i] = readincr_v<4>(h_vector_in);
            result[i] = aie::add(result[i], h[i]);
        }
        // chess_separator_scheduler();
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
            {
            for (int j = 0; j < VECTOR_LANES; j++) chess_loop_count(VECTOR_LANES)
                {
                // The index is : base id passed + the DIST offset + the current VECTOR LANE
                int idx = id + i*VECTOR_LANES + j;
                writeHeader(out,pktType,j);
                writeincr(out, idx);
                writeincr(out, result[i].get(j), true);
            }
        }
        chess_separator_scheduler(VECTOR_LANES);
    }
}