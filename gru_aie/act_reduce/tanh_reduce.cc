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
                    const unsigned int (&identifier)

){  
    alignas(32) float res;
    alignas(32) float wrx[DIST_COEFF], urx[DIST_COEFF];
    
    static constexpr float tanh_thresh = 3.0;
    static constexpr float tanh_m_coeff = 4096.0 / 6.0;

    static const unsigned int pktType = 0;
    static const unsigned int ID = getPacketid(out,0); //for output pktstream

    for(;;){

        for (int i = 0; i < VECTOR_LANES ; i++)
        {   wrx[i] = readincr(x_in);
            urx[i] = readincr(h_in);
        }

        for (int i = 0; i < DIST_COEFF; i++)
        {   
            res = bias[i]; 
            res += aie::reduce_add(aie::load_v<4>((float*)&wrx)); 
            res += aie::reduce_add(aie::load_v<4>((float*)&urx));
 
            if (res <= - tanh_thresh){
                writeHeader(out,pktType,ID); //Generate header for output
                writeincr(out, identifier);
                writeincr(out, 0, true); //TLAST=true for last word

            } else if (res >= tanh_thresh){
                writeHeader(out,pktType,ID);
                writeincr(out, identifier);
                writeincr(out, 1, true);

            } else {
                int index = int((res + tanh_thresh)*tanh_m_coeff) ; // Eventually change these values to config
                writeHeader(out,pktType,ID);
                writeincr(out, identifier);
                writeincr(out,tan_h[index],true);
            }
        }
    }
}