#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "mat_vec_mul.h"

void mat_vec_mul(   adf::input_circular_buffer           <float,adf::extents<X_SIZE>>      & __restrict in,
                    adf::output_async_circular_buffer    <float,adf::extents<DIST_COEFF*VECTOR_SIZE>> & __restrict out,
                    const float (&weights)[X_SIZE*DIST_COEFF]
){
    auto pin = aie::begin_vector_circular<VECTOR_SIZE>(in);
    auto pout = aie::begin_vector_circular<VECTOR_SIZE>(out);

    alignas(32) aie::accum<accfloat, 8> accum;
    alignas(32) aie::vector<float, 8> x_input[X_SIZE/VECTOR_SIZE];

    for (;;){
        // Read the input and keep it
        for (int i = 0; i < X_SIZE/VECTOR_SIZE; i++){ x_input[i] = *pin++;}

        for (int i = 0; i < DIST_COEFF; i++)
        {   accum.from_vector(aie::zeros<float, 8>());
            for (int j = 0; j < X_SIZE/VECTOR_SIZE; j++)
                {
                accum = aie::mac(accum, x_input[j], aie::load_v<8>((float*)&weights[i*X_SIZE + VECTOR_SIZE*j]));
            };

            out.acquire();
            *pout++ = accum;
            out.release();

        };
    };
};
