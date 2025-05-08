#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "tanh_reduce.h"
#include "act_funcs_luts/tanh.h"
#include "../config.h"

void tanh_reduce(   input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF],
                    const int (&identifier)

){  
    alignas(32) float res;
    
    static constexpr float lut_max_idx = lut_size - 1.0;
    static constexpr float tanh_m_coeff = lut_max_idx/ (2.0 * tanh_thresh);

    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); //for output pktstream

    for (;;){
        chess_separator_scheduler();
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
        {
            res = bias[i];
            res += readincr(x_in);
            res += readincr(h_in);

            if (res <= - tanh_thresh){
                writeHeader(out,pktType,ID); //Generate header for output
                writeincr(out, identifier + i);
                writeincr(out, -1, true); //TLAST=true for last word
            } else if (res >= tanh_thresh){
                writeHeader(out,pktType,ID);
                writeincr(out, identifier + i);
                writeincr(out, 1, true);
            } else {
                int index = int((res + tanh_thresh)*tanh_m_coeff) ; // Eventually change these values to config
                writeHeader(out,pktType,ID);
                writeincr(out, identifier + i);
                writeincr(out,tan_h[index],true);
            }
        }
        chess_separator_scheduler(3);
    }
}
