#include <array>
#include "gru_atom.h"

gru_atom g0;

int main(int argc, char ** argv)
{   // Array to simulate the RTPs
    // std::array<float,DIST_COEFF*X_VECTOR_SIZE> test_params;
    // for (int i = 0; i < DIST_COEFF*X_VECTOR_SIZE; i++){test_params[i] = i;}

    g0.init();
    // Pass all the RTPs
    // for (int i = 0; i < H_VECTOR_SIZE/DIST_COEFF; i++) {g0.update(g0.W_weights[i], test_params.data(), DIST_COEFF*X_VECTOR_SIZE);}
    g0.run(1);
    g0.end();
    return 0;
}