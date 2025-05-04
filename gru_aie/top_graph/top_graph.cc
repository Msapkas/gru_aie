#include "top_graph.h"
#include "../utils/utils.h"
#include "config.h"

// Instantiate graph
top_graph top_graph;

int main(int argc, char ** argv){

    // Declare the arrays

    float value;

    // ----------------- GRU 

    float h_init_params [H_VECTOR_SIZE];
    int ID_param [NKERNELS];

    // ---------------- GRU gates

    float R_Wx_params[H_VECTOR_SIZE*X_VECTOR_SIZE];
    float R_Uh_params[H_VECTOR_SIZE*H_VECTOR_SIZE];
    float R_b_params[H_VECTOR_SIZE];

    float Z_Wx_params[H_VECTOR_SIZE*X_VECTOR_SIZE];
    float Z_Uh_params[H_VECTOR_SIZE*H_VECTOR_SIZE];
    float Z_b_params[H_VECTOR_SIZE];

    float Chsg_Wx_params[H_VECTOR_SIZE*X_VECTOR_SIZE];
    float Chsg_Uh_params[H_VECTOR_SIZE*H_VECTOR_SIZE];
    float Chsg_b_params[H_VECTOR_SIZE];

    // ------------------ LINEAR LAYERs

    float L0_params[H_VECTOR_SIZE*output_dims_0];
    float b0_params[output_dims_0];
    float L1_params[output_dims_0];
    float b1_params[1];

    // Pass from txt to array the floating point values

    for (int i = 0; i < H_VECTOR_SIZE; i++){h_init_params[i] = 0.0;}
    for (int i = 0; i < NKERNELS; i++){ID_param[i] = i*DIST_COEFF;}

    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_relevance_wxh.txt", R_Wx_params, H_VECTOR_SIZE*X_VECTOR_SIZE);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_relevance_whh.txt", R_Uh_params, H_VECTOR_SIZE*H_VECTOR_SIZE);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_relevance_b.txt", R_b_params, H_VECTOR_SIZE);

    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_update_wxh.txt", Z_Wx_params, H_VECTOR_SIZE*X_VECTOR_SIZE);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_update_whh.txt", Z_Uh_params, H_VECTOR_SIZE*H_VECTOR_SIZE);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_update_b.txt", Z_b_params, H_VECTOR_SIZE);

    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_candidate_wxh.txt", Chsg_Wx_params, H_VECTOR_SIZE*X_VECTOR_SIZE);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_candidate_whh.txt", Chsg_Uh_params, H_VECTOR_SIZE*H_VECTOR_SIZE);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/gru_candidate_b.txt", Chsg_b_params, H_VECTOR_SIZE);

    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/linear1_weight.txt", L0_params, H_VECTOR_SIZE*output_dims_0);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/linear1_bias.txt", b0_params, output_dims_0);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/linear2_weight.txt", L1_params, output_dims_0);
    loadFloatsFromFile("/home/sapkas/GRU/gru_aie/model_parameters/linear2_bias.txt", b1_params, 1);

    // Init
    top_graph.init();
    //test_bench.wait(10);

    // ----------------- GRU 

    for (int i = 0; i < NKERNELS; i++) {
        top_graph.update(top_graph.gru_sub_graph.r_identifier[i], &ID_param[i], 1);
        top_graph.update(top_graph.gru_sub_graph.r_hidden_initialization[i], h_init_params, H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Wr_params[i], &R_Wx_params[i*DIST_COEFF*X_VECTOR_SIZE], DIST_COEFF*X_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Ur_params[i], &R_Uh_params[i*DIST_COEFF*H_VECTOR_SIZE], DIST_COEFF*H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.br_params[i], &R_b_params[i*DIST_COEFF], DIST_COEFF);

        top_graph.update(top_graph.gru_sub_graph.z_identifier[i], &ID_param[i], 1);
        top_graph.update(top_graph.gru_sub_graph.z_hidden_initialization[i], h_init_params, H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Wz_params[i], &Z_Wx_params[i*DIST_COEFF*X_VECTOR_SIZE], DIST_COEFF*X_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Uz_params[i], &Z_Uh_params[i*DIST_COEFF*H_VECTOR_SIZE], DIST_COEFF*H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.bz_params[i], &Z_b_params[i*DIST_COEFF], DIST_COEFF);

        top_graph.update(top_graph.gru_sub_graph.chsg_identifier[i], &ID_param[i], 1);
        top_graph.update(top_graph.gru_sub_graph.chsg_hidden_initialization[i], h_init_params, H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Wh_params[i], &Chsg_Wx_params[i*DIST_COEFF*X_VECTOR_SIZE], DIST_COEFF*X_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.Uh_params[i], &Chsg_Uh_params[i*DIST_COEFF*H_VECTOR_SIZE], DIST_COEFF*H_VECTOR_SIZE);
        top_graph.update(top_graph.gru_sub_graph.bh_params[i], &Chsg_b_params[i*DIST_COEFF], DIST_COEFF);
    }
    top_graph.update(top_graph.gru_sub_graph.new_hidden_state_gate_hidden_initialization, h_init_params, H_VECTOR_SIZE);

    // ------------------ LINEAR LAYER

    top_graph.update(top_graph.linear_layer_sub_graph.linear_params_0, L0_params, H_VECTOR_SIZE*output_dims_0);
    top_graph.update(top_graph.linear_layer_sub_graph.bias_0, b0_params, output_dims_0);
    top_graph.update(top_graph.linear_layer_sub_graph.linear_params_1, L1_params, output_dims_0);
    top_graph.update(top_graph.linear_layer_sub_graph.bias_1, b1_params, 1);

    // -------- RTPs Passed

    top_graph.run(1);
    top_graph.end();
    return 0;
}