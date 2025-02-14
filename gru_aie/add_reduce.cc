#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "add_reduce.h"
#include "act_funcs_luts/sigm.h"
#include "./config.h"

void add_reduce(adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict x_in,
                adf::input_circular_buffer<float,adf::extents<H_VECTOR_SIZE*VECTOR_LANES>>& __restrict h_in,
                adf::output_pktstream *out,
                const float (&bias)[MULT_DIST_COEFF]

){  auto x_pin = aie::begin_vector_circular<VECTOR_LANES>(x_in);
    auto h_pin = aie::begin_vector_circular<VECTOR_LANES>(h_in);

    float gate_result[MULT_DIST_COEFF]; 

    uint32 ID=getPacketid(out,0);//for output pktstream
    static const unsigned int pktType=0;
	writeHeader(out,pktType,ID); //Generate header for output

    for(;;){
        for (int i = 0; i < MULT_DIST_COEFF; i++){
            for (int i = 0; i < H_VECTOR_SIZE/VECTOR_LANES){
                gate_result[i] = aie::reduce_add(*x_pin++) + aie::reduce_add(*h_pin++) + bias[i];
            }
        }

        for (int i = 0; i < MULT_DIST_COEFF; i++)chess_unrollloop_(*){
            writeincr(out, sigm[gate_result[i]], i==MULT_DIST_COEFF);
        }
    }
}