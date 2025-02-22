#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid_reduce.h"
#include "act_funcs_luts/sigm.h"
#include "config.h"

void sigmoid_reduce(adf::input_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>, adf::margin<VECTOR_LANES>>& __restrict x_in,
                    adf::input_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>, adf::margin<VECTOR_LANES>>& __restrict h_in,
                    output_pktstream *out,
                    const float (&bias)[DIST_COEFF]

){  auto x_pin = aie::begin_vector_circular<VECTOR_LANES>(x_in);
    auto h_pin = aie::begin_vector_circular<VECTOR_LANES>(h_in);

    float res[DIST_COEFF];
    float gate_result[DIST_COEFF];
    
    static const unsigned int pktType = 0;
    uint32 ID=getPacketid(out,0);//for output pktstream

    for(;;){
		
        for (int i = 0; i < DIST_COEFF; i++){
            aie::vector<float, 8> wrx[i] = *x_pin++;
            aie::vector<float, 8> urx[i] = *h_pin++; 
        }
        
        for (int i = 0; i < DIST_COEFF; i++){
            res = bias[i];
            res[i] += aie::reduce_add(wrx[i]);
            res[i] +=aie::reduce_add(urx[i]);

            if (res <= -SIGMOID_THR){
                writeHeader(out,pktType,ID); //Generate header for output
                writeincr(out,0,i==DIST_COEFF-1); //TLAST=1 for last word
            } else if (res[i] >= SIGMOID_THR){
                writeHeader(out,pktType,ID); //Generate header for output
                writeincr(out,1,i==DIST_COEFF-1); //TLAST=1 for last word
            } else {
                size_t index = static_cast<size_t>((res[i] + SIGMOID_THR) / LUT_SIZE);
                writeHeader(out,pktType,ID); //Generate header for output
                writeincr(out,sigm[index],i==DIST_COEFF-1); //TLAST=1 for last word
            }
        }
    }
}