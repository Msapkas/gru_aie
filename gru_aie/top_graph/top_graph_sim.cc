#include "top_graph.h"

// Instantiate graph
top_graph top_graph;

int main(int argc, char ** argv){

    // RTP Simulation

    // ----------------- GRU 

    float h_init_test_params [H_VECTOR_SIZE];
    int ID_param [NKERNELS];
    float W_test_params[DIST_COEFF*X_VECTOR_SIZE];
    float U_test_params[DIST_COEFF*H_VECTOR_SIZE];
    float b_test_params[H_VECTOR_SIZE];

    for (int i = 0; i < H_VECTOR_SIZE; i++){h_init_test_params[i] = 0.01*i;}
    for (int i = 0; i < NKERNELS; i++){ID_param[i] = i*DIST_COEFF;}
    for (int i = 0; i < DIST_COEFF*X_VECTOR_SIZE; i++){W_test_params[i] = 0.001*i;}
    for (int i = 0; i < H_VECTOR_SIZE; i++){b_test_params[i] = 0.02*i;}
    for (int i = 0; i < DIST_COEFF*H_VECTOR_SIZE; i++){U_test_params[i] = 0.005*i;}

    // ------------------ LINEAR LAYER

    float L_test_params_0[H_VECTOR_SIZE*output_dims_0];
    float b_test_params_0[output_dims_0];
    float L_test_params_1[output_dims_0];
    
    for (int i = 0; i < H_VECTOR_SIZE*output_dims_0; i++){L_test_params_0[i] = 1;}
    for (int i = 0; i < output_dims_0; i++){b_test_params_0[i] = 1;}

    for (int i = 0; i < output_dims_0; i++){L_test_params_1[i] = 0.01*i;}
    float b_test_params_1 = 0.1;

    // Init
    top_graph.init();
    //test_bench.wait(10);

    // ----------------- GRU 

    for (int i = 0; i < NKERNELS; i++) {
        top_graph.update(top_graph.gru_sub_graph.r_identifier[i], &ID_param[i], 1);
        top_graph.update(top_graph.gru_sub_graph.Ur_params[i], U_test_params, DIST_COEFF*H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.r_hidden_initialization[i], h_init_test_params, H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Wr_params[i], W_test_params, DIST_COEFF*X_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.br_params[i], &b_test_params[i*DIST_COEFF], DIST_COEFF);

        top_graph.update(top_graph.gru_sub_graph.z_identifier[i], &ID_param[i], 1);
        top_graph.update(top_graph.gru_sub_graph.z_hidden_initialization[i], h_init_test_params, H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Wz_params[i], W_test_params, DIST_COEFF*X_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Uz_params[i], U_test_params, DIST_COEFF*H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.bz_params[i], &b_test_params[i*DIST_COEFF], DIST_COEFF);

        top_graph.update(top_graph.gru_sub_graph.chsg_identifier[i], &ID_param[i], 1);
        top_graph.update(top_graph.gru_sub_graph.chsg_hidden_initialization[i], h_init_test_params, H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Wh_params[i], W_test_params, DIST_COEFF*X_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Uh_params[i], U_test_params, DIST_COEFF*H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.bh_params[i], &b_test_params[i*DIST_COEFF], DIST_COEFF);
    }
    top_graph.update(top_graph.gru_sub_graph.new_hidden_state_gate_hidden_initialization, h_init_test_params, H_VECTOR_SIZE);

    // ------------------ LINEAR LAYER

    top_graph.update(top_graph.linear_layer_sub_graph.linear_params_0, L_test_params_0, H_VECTOR_SIZE*output_dims_0);
    top_graph.update(top_graph.linear_layer_sub_graph.bias_0, b_test_params_0, output_dims_0);
    top_graph.update(top_graph.linear_layer_sub_graph.linear_params_1, L_test_params_1, output_dims_0);
    top_graph.update(top_graph.linear_layer_sub_graph.bias_1, b_test_params_1);

    // -------- RTP Passed

    top_graph.run(1);
    top_graph.end();
    return 0;
}