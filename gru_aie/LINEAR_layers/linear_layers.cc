#include "linear_layers.h"
#include "../config.h"

linear_layers linear_layers;

int main(int argc, char ** argv)
{   
    // Array to simulate the RTPs
    // float L_test_params_0[H_VECTOR_SIZE*output_dims_0];
    // float b_test_params_0[output_dims_0];
    // float L_test_params_1[output_dims_0];
    
    // for (int i = 0; i < H_VECTOR_SIZE*output_dims_0; i++){L_test_params_0[i] = 1;}
    // for (int i = 0; i < output_dims_0; i++){b_test_params_0[i] = 1;}

    // for (int i = 0; i < output_dims_0; i++){L_test_params_1[i] = 0.01*i;}
    // float b_test_params_1 = 0.1;

    linear_layers.init();

    // linear_layers.update(linear_layers.linear_params_0, L_test_params_0, H_VECTOR_SIZE*output_dims_0);
    // linear_layers.update(linear_layers.bias_0, b_test_params_0, output_dims_0);
    // linear_layers.update(linear_layers.linear_params_1, L_test_params_1, output_dims_0);
    // linear_layers.update(linear_layers.bias_1, b_test_params_1);

    linear_layers.run(-1);
    linear_layers.end();
    return 0;
}