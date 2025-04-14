#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid.h"
#include "act_funcs_luts/sigm.h"
#include "../config.h"

void sigmoid(   input_pktstream * in,
                output_pktstream * out

){  // This value have to do with the threshold of the input.
    static constexpr float sigm_thresh = 6.0;
    // This value pre-calculates the division needed to scale the input value between 0 - 4096, which are the indexes of the LUT
    // by precalculating this number we skip cycle consuming division and do a multiplication instead!
    static constexpr float sigm_m_coeff = 4096.0 / 12.0;

    // Some header values
    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); // for output pktstream the ID of a merge is 0

    float input[DIST_COEFF];

    for (;;){
        for (int i = 0; i < DIST_COEFF; i++ ) chess_loop_count(DIST_COEFF)
            {
            //
            readincr(in); // read header and discard
            int idx = int(readincr(in));  // read ID
            unsigned int pkt_input = readincr(in);
            unsigned int* src = (unsigned int*)& pkt_input;
            float* dest = (float*) src;
            input[i] = *dest;
            //
            if (input[i] <= - sigm_thresh){
                writeHeader(out,pktType,ID);    // Generate header for output
                writeincr(out, idx); // Write index 
                writeincr(out, 0, true);        // TLAST=true for last word

            } else if (input[i] >= sigm_thresh){
                writeHeader(out,pktType,ID);
                writeincr(out, idx);
                writeincr(out, 1, true);

            } else {
                // if you do: employ the LUT
                int index = int((input[i] + sigm_thresh)*sigm_m_coeff) ;
                writeHeader(out,pktType,ID);
                writeincr(out, idx);
                writeincr(out,sigm[index],true);
            }
        }
    }
}
