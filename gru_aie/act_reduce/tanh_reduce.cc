#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "tanh_reduce.h"
#include "act_funcs_luts/tanh.h"
#include "./config.h"

void tanh_reduce(adf::input_async_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict x_in,
                    adf::input_async_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF]

){  
    auto p_xin = aie::begin_vector_circular<VECTOR_LANES>(x_in);
    auto p_hin = aie::begin_vector_circular<VECTOR_LANES>(h_in);

    float res[DIST_COEFF];
    aie::vector<float, 8> wrx[DIST_COEFF], urx[DIST_COEFF];
    
    static constexpr float tn_m_coeff = 4096.0 / 6.0;

    static const unsigned int pktType = 0;
    uint32 ID=getPacketid(out,0);//for output pktstream

    for(;;){
        
        for (int i = 0; i < DIST_COEFF; i++){
            x_in.acquire();
            wrx[i] = *p_xin++;
            x_in.release();
            h_in.acquire();
            urx[i] = *p_hin++;
            h_in.release();

            res[i] = bias[i] + aie::reduce_add(wrx[i]) + aie::reduce_add(urx[i]);
        }

        for (int i = 0; i < DIST_COEFF; i++){
            if (res[i] <= -TANH_THR){
                writeHeader(out,pktType,ID); //Generate header for output
                writeincr(out, 0, true); //TLAST=1 for last word
            } else if (res[i] >= TANH_THR){
                writeHeader(out,pktType,ID);
                writeincr(out, 1, true);
            } else {
                size_t index = static_cast<size_t>(res[i] + 3.0)*tn_m_coeff; // Eventually change these values to config
                writeHeader(out,pktType,ID);
                writeincr(out, tan_h[index], true);
            }
        }
    }
}