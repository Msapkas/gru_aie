#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_input_vec_mul.h"
#include "./config.h"

// template<int VECTOR_SIZE, int VECTOR_LANES, int MULT_DIST_COEFF> 
void mat_input_vec_mul(     adf::input_circular_buffer           <float,adf::extents<X_VECTOR_SIZE>>              & __restrict in,
                            adf::output_async_circular_buffer    <float,adf::extents<MULT_DIST_COEFF*VECTOR_LANES>>    & __restrict out,
                            const float (&weights)[X_VECTOR_SIZE*MULT_DIST_COEFF]

){  auto pin = aie::begin_vector_circular<VECTOR_LANES>(in);
    auto pout = aie::begin_vector_circular<VECTOR_LANES>(out);

    alignas(32) aie::accum<accfloat, 8> accum;
    alignas(32) aie::vector<float, 8> x_input[X_VECTOR_SIZE/VECTOR_LANES];

    for (;;){
        // Read the input and keep it
        for (int i = 0; i < X_VECTOR_SIZE/VECTOR_LANES; i++){ x_input[i] = *pin++;}

        for (int i = 0; i < MULT_DIST_COEFF; i++)
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
