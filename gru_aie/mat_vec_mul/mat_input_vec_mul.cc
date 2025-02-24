#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "adf/io_buffer/io_buffer_types.h"
#include "mat_input_vec_mul.h"
#include "../config.h"

void mat_input_vec_mul( input_stream<float> * __restrict in,
                        adf::output_async_circular_buffer <float,adf::extents<DIST_COEFF*VECTOR_LANES>> & __restrict out,
                        const float (&weights)[X_VECTOR_SIZE*DIST_COEFF]

){  
    auto pout = aie::begin_vector_circular<VECTOR_LANES>(out);

    alignas(32) aie::accum<accfloat, 8> accum;
    alignas(32) aie::vector<float, 8> x_input[X_VECTOR_SIZE/VECTOR_LANES];

    for (;;){

        // Read the input and keep it
        for (int i = 0; i < X_VECTOR_SIZE/VECTOR_LANES; i++){ x_input[i] = readincr_v<8>(in);}

        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 8>());
            for (int j = 0; j < X_VECTOR_SIZE/VECTOR_LANES; j++)
                {
                accum = aie::mac(accum, x_input[j], aie::load_v<8>((float*)&weights[i*X_VECTOR_SIZE + VECTOR_LANES*j]));
            }

        out.acquire();
        *pout++ = accum;
        out.release();
        
        }
    }
}
