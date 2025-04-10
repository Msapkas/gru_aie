#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "adder_tanh.h"
#include "act_funcs_luts/tanh.h"
#include "../config.h"

void adder_tanh(   input_stream<float> * __restrict x_in,
                    input_stream<float> * __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF*VECTOR_LANES],
                    const int (&identifier)

){  
    alignas(32) aie::vector<float, VECTOR_LANES> wrx[DIST_COEFF], urx[DIST_COEFF], res[DIST_COEFF];
    
    static constexpr float tanh_thresh = 3.0;
    static constexpr float tanh_m_coeff = 4096.0 / 6.0;

    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); //for output pktstream

    /// infinite loop
    for (;;){
        // Accept the MACed results
        for (int i = 0; i < DIST_COEFF ; i++) chess_loop_count(DIST_COEFF)
            {
            wrx[i] = readincr_v<4>(x_in);
            urx[i] = readincr_v<4>(h_in);
        }
        chess_separator_scheduler();
        // add the vectors and the bias element wise
        for (int i = 0; i < DIST_COEFF; i++) chess_loop_count(DIST_COEFF)
            {   
            res[i] = aie::add(wrx[i], bias[i]);
            res[i] = aie::add(res[i], urx[i]);
        }
        for (int dist = 0; dist < DIST_COEFF; dist++ ) chess_loop_count(DIST_COEFF) chess_unroll_loop(*)
            {
            for (int i = 0; i < VECTOR_LANES; i++) chess_loop_count(VECTOR_LANES) chess_unroll_loop(*)
                {
                int curr_id = identifier + DIST_COEFF*i + i;
                // once calculated the result check if you fall between the threshold sigm_thresh
                if (res[dist][i] <= - tanh_thresh){
                    writeHeader(out,pktType,ID);    // Generate header for output
                    writeincr(out, curr_id); // Write index 
                    writeincr(out, 0, true);        // TLAST=true for last word
                } else if (res[dist][i] >= tanh_thresh){
                    writeHeader(out,pktType,ID);
                    writeincr(out, curr_id);
                    writeincr(out, 1, true);
                } else {
                    // if you do: employ the LUT
                    int index = int((res[dist][i] + tanh_thresh)*tanh_m_coeff) ; // Eventually change these values to config
                    writeHeader(out,pktType,ID);
                    writeincr(out, curr_id);
                    writeincr(out,tan_h[index],true);
                }
            }
        }
        chess_separator_scheduler();
    }
}
