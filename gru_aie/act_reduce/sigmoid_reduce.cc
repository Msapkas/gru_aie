#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid_reduce.h"
#include "act_funcs_luts/sigm.h"
#include "../config.h"

void sigmoid_reduce(adf::input_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict x_in,
                    adf::input_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict h_in,
                    output_pktstream * out,
                    const float (&bias)[DIST_COEFF]

){  
    auto p_xin = aie::begin_vector_random_circular<VECTOR_LANES>(x_in);
    auto p_hin = aie::begin_vector_random_circular<VECTOR_LANES>(h_in);

    float res;
    float gate_result[DIST_COEFF];

    aie::vector<float, 8> wrx, urx;
    
    static const unsigned int pktType = 0;
    uint32 ID=getPacketid(out,0);//for output pktstream

    for(;;){
        
        for (int i = 0; i < DIST_COEFF; i++){
            x_in.acquire();
            wrx = *p_xin++;
            x_in.release();
            h_in.acquire();
            urx = *p_hin++;
            h_in.release();

            res = bias[i];
            res += aie::reduce_add(wrx);
            res += aie::reduce_add(urx);

            if (res <= -SIGMOID_THR){

                // writeHeader(out,pktType,ID); //Generate header for output
                // writeincr(out,0,true); //TLAST=1 for last word
                size_t index = static_cast<size_t>((res + SIGMOID_THR) / LUT_SIZE);
                writeHeader(out,pktType,ID);
                writeincr(out,sigm[index],true);

            } else if (res >= SIGMOID_THR){

                // writeHeader(out,pktType,ID);
                // writeincr(out,1,true);
                size_t index = static_cast<size_t>((res + SIGMOID_THR) / LUT_SIZE);
                writeHeader(out,pktType,ID);
                writeincr(out,sigm[index],true);

            } else {

                size_t index = static_cast<size_t>((res + SIGMOID_THR) / LUT_SIZE);
                writeHeader(out,pktType,ID);
                writeincr(out,sigm[index],true);

            }
        }
    }
}