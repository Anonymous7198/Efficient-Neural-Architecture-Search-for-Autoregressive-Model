# Efficient Neural Architecture Search for Neural Autoregressive Model

## CIFAR-10

To run ENAS experiments, please use the following scripts:
```
./scripts/cifar10_micro_search.sh

```
The cell contains of B blocks, each block can be described with: 
```
hidden[x_id], hidden[y_id], ops[x_op], ops[y_op], act[x_act], act[y_act], comb[comb_id]

```
Search space (note that all conv + self attention ops are causal and can be implemented with shifting and padding)
```
hidden are taken from two previous cells outputs

ops are taken from:
  3x3 causal conv                       3x3 causal group conv
  5x5 causal conv                       5x5 causal group conv
  3x3 causal depthwise conv             causal self-attention 
  5x5 causal depthwise conv             identity

act are taken from:
  swish activation                      sigmoid activation
  relu activation                       tanh activation
  leaky relu activation                 identity

comb are taken from:
  add, element wise mul and concat
```
For CIFAR-10, reward for our controller is c / negative_log_likelihood (c is constant)

Experiments results: will be updated very soon
