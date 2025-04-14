#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "hyper_tan.h"
#include "act_funcs_luts/tanh.h"
#include "../config.h"

void hyper_tan(  input_pktstream * in,
            output_pktstream * out

){  
    static constexpr float tanh_thresh = 3.0;
    static constexpr float tanh_m_coeff = 4096.0 / 6.0;

    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); //for output pktstream

    float input[DIST_COEFF];

    for (;;){
        // Accept input (v4)
        for (int i = 0; i < DIST_COEFF ; i++) chess_loop_count(DIST_COEFF)
            {
            //
            readincr(in); // read header and discard
            int idx = int(readincr(in));  // read ID
            unsigned int pkt_input = readincr(in);
            unsigned int* src = (unsigned int*)& pkt_input;
            float* dest = (float*) src;
            input[i] = *dest;
            //
            if (input[i] <= - tanh_thresh){
                writeHeader(out,pktType,ID);    // Generate header for output
                writeincr(out, idx); // Write index 
                writeincr(out, 0, true);        // TLAST=true for last wor

            } else if (input[i] >= tanh_thresh){
                writeHeader(out,pktType,ID);
                writeincr(out, idx);
                writeincr(out, 1, true);

            } else {
                // if you do: employ the LUT
                int index = int((input[i] + tanh_thresh)*tanh_m_coeff) ; // Eventually change these values to config
                writeHeader(out,pktType,ID);
                writeincr(out, idx);
                writeincr(out,tan_h[index],true);
            }
        }
    }
}
