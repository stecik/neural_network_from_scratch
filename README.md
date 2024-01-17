# Neural network from scratch
Implementation of neural networks from scratch - no machine learning libraries, only numpy and math. This project offers a deep dive into the core principles of neural computation through a hands-on, educational approach. The main purpose is to understand in low level detail how backpropagation and neural networks work. As the code does not use tensors and therefore is not optimized to run in parallel on GPU, it might be quite slow for larger networks.
## Implemented and tested:
1) Activation functions
   - Sigmoid
   - ReLU
   - Leaky ReLU
   - ELU
   - Softmax (for output layer only)
   - Tanh
   - Softplus
   - Swish (to be added)
2) Loss functions
   - MSE
   - Categorical cross entropy
3) Layers types
   - Dense layer
   - CNN (to be added)
   - LSTM (to be added)
4) Forward step and backpropagation
5) Adaptive learning rate
6) Data normalization - to values in interval [0,1] 

## Network performance
### MNIST Digits dataset:
- Test accuracy: 88.92%
