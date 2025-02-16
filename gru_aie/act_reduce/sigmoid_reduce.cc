#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "sigmoid_reduce.h"
#include "act_funcs_luts/sigm.h"
#include "config.h"

void sigmoid_reduce(adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<DIST_COEFF*VECTOR_LANES>>& __restrict h_in,
                output_pktstream *out,
                const float (&bias)[DIST_COEFF]

){  auto x_pin = aie::begin_vector_circular<VECTOR_LANES>(x_in);
    auto h_pin = aie::begin_vector_circular<VECTOR_LANES>(h_in);

    float res[DIST_COEFF];
    float gate_result[DIST_COEFF];

    for(;;){

        for (int i = 0; i < DIST_COEFF; i++){
            for (int j = 0; i < H_VECTOR_SIZE/VECTOR_LANES; j++){

                // aie::vector<float, VECTOR_LANES> x_data = *x_pin++;
                // aie::vector<float, VECTOR_LANES> h_data = *h_pin++;
                // res[j] = aie::reduce_add(x_data) + aie::reduce_add(h_data) + bias[j];

                res[j] = aie::reduce_add(*x_pin++) + aie::reduce_add(*h_pin++) + bias[j];
            }
        }
        
        for (int i = 0; i < DIST_COEFF; i++)chess_unroll_loop(*){
            if (res[i] < -SIGMOID_THR){
                gate_result[i] = 0;
            } else if (res[i] > SIGMOID_THR){
                gate_result[i] = 1;
            } else {
                gate_result[i] = sigm[res[i]];
            }
        }

        static const unsigned int pktType = 0;
        uint32 ID=getPacketid(out,0);//for output pktstream
		writeHeader(out,pktType,ID); //Generate header for output

        for (int i = 0; i < DIST_COEFF; i++){
            writeincr(out,gate_result[i],i==DIST_COEFF-1); //TLAST=1 for last word
        }

    }
}