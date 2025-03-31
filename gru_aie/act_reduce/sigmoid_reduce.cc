#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid_reduce.h"
#include "act_funcs_luts/sigm.h"
#include "../config.h"

void sigmoid_reduce(input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF],
                    const int (&identifier)

){  
    alignas(32) float res;
    alignas(32) aie::vector<float, VECTOR_LANES> wrx[DIST_COEFF], urx[DIST_COEFF];
    
    // This value have to do with the threshold of the input.
    static constexpr float sigm_thresh = 6.0;
    // This value pre-calculates the division needed to scale the input value between 0 - 4096, which are the indexes of the LUT
    // by precalculating this number we skip cycle consuming division and do a multiplication instead!
    static constexpr float sigm_m_coeff = 4096.0 / 12.0;

    // Some header values
    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); // for output pktstream the ID of a merge is 0

    // infinite loop
    for (;;){
        chess_separator_scheduler();
        // Accept the last mac of a row
        for (int i = 0; i < DIST_COEFF ; i++) chess_loop_count(DIST_COEFF)
            {
            wrx[i] = readincr_v<4>(x_in);
            urx[i] = readincr_v<4>(h_in);
        }
        chess_separator_scheduler();
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
            {   
            // add the bias and reduce the vectors
            res = bias[i];
            res += aie::reduce_add(wrx[i]);
            res += aie::reduce_add(urx[i]);

            // once calculated the result check if you fall between the threshold sigm_thresh
            if (res <= - sigm_thresh){
                writeHeader(out,pktType,ID);    // Generate header for output
                writeincr(out, identifier + i); // Write index 
                writeincr(out, 0, true);        // TLAST=true for last word
            } else if (res >= sigm_thresh){
                writeHeader(out,pktType,ID);
                writeincr(out, identifier + i);
                writeincr(out, 1, true);
            } else {
                // if you do: employ the LUT
                int index = int((res + sigm_thresh)*sigm_m_coeff) ;
                writeHeader(out,pktType,ID);
                writeincr(out, identifier + i);
                writeincr(out,sigm[index],true);
            }
        }
        chess_separator_scheduler();
    }
}
