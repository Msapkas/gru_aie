# gru_aie repo
A low latency implementation of the Gated Recurrent Unit on the Versal AI Engines

Different implementations of the model can be found by switching branches. Here are the branches and their descriptions:

- gru_act_compute_linear : The GRU model, using the MAC rows method with kernels that **compute** the activation functions as splines. A linear layer is connected at the end. (most recent)
- gru_reducers_w_linear : The GRU model, using MAC rows method with kernels that implement **Look Up Tables** to calculate the activation functions. A linear layer is connected at the end. (mosts recent and numerically validated)
- macs_implementation: The GRU model, using the MAC **columns** method and kernels that implement **Look Up Tables** for activation functions.
- master_aggregator: An old GRU model, that is not numericaly tested. In this branch I am testing an idea to double the aggregation limitation from 32 to 64 merges.
- reducers_implementation: An old GRU model that is using the MAC rows method.
