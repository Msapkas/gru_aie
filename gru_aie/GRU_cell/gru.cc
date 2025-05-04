#include "gru.h"
#include "../config.h"

// Instantiate graph
gru gru_graph;

int main(int argc, char ** argv){

    // Array to simulate the RTPs
    // float h_init_test_params [H_VECTOR_SIZE];
    // int ID_param [NKERNELS];
    // float W_test_params[DIST_COEFF*X_VECTOR_SIZE];
    // float U_test_params[DIST_COEFF*H_VECTOR_SIZE];
    // float b_test_params[H_VECTOR_SIZE];

    // for (int i = 0; i < H_VECTOR_SIZE; i++){h_init_test_params[i] = 0.01*i;}
    // for (int i = 0; i < NKERNELS; i++){ID_param[i] = i*DIST_COEFF;}
    // for (int i = 0; i < DIST_COEFF*X_VECTOR_SIZE; i++){W_test_params[i] = 0.001*i;}
    // for (int i = 0; i < H_VECTOR_SIZE; i++){b_test_params[i] = 0.02*i;}
    // for (int i = 0; i < DIST_COEFF*H_VECTOR_SIZE; i++){U_test_params[i] = 0.005*i;}
    
    gru_graph.init();
    // Pass all the RTPs
    // for (int i = 0; i < NKERNELS; i++) {
    //     gru_graph.update(gru_graph.r_identifier[i], &ID_param[i], 1);
    //     gru_graph.update(gru_graph.Ur_params[i], U_test_params, DIST_COEFF*H_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.r_hidden_initialization[i], h_init_test_params, H_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.Wr_params[i], W_test_params, DIST_COEFF*X_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.br_params[i], &b_test_params[i*DIST_COEFF], DIST_COEFF);

    //     gru_graph.update(gru_graph.z_identifier[i], &ID_param[i], 1);
    //     gru_graph.update(gru_graph.z_hidden_initialization[i], h_init_test_params, H_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.Wz_params[i], W_test_params, DIST_COEFF*X_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.Uz_params[i], U_test_params, DIST_COEFF*H_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.bz_params[i], &b_test_params[i*DIST_COEFF], DIST_COEFF);

    //     gru_graph.update(gru_graph.chsg_identifier[i], &ID_param[i], 1);
    //     gru_graph.update(gru_graph.chsg_hidden_initialization[i], h_init_test_params, H_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.Wh_params[i], W_test_params, DIST_COEFF*X_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.Uh_params[i], U_test_params, DIST_COEFF*H_VECTOR_SIZE);
    //     gru_graph.update(gru_graph.bh_params[i], &b_test_params[i*DIST_COEFF], DIST_COEFF);
    // }
    // gru_graph.update(gru_graph.new_hidden_state_gate_hidden_initialization, h_init_test_params, H_VECTOR_SIZE);
    //
    // gru_graph.wait(10);
    gru_graph.run(-1);
    gru_graph.end();
    return 0;

};