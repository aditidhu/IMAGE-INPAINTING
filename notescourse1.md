# **Image Inpainting**:
# Course 1:
## Week 1:
# Neural Networks And Deep Learning
  > Consists of neurons which bssically give an output for given input 
  > Neural networks are formed by comination of multiple neuron layers
  > Every Input is interconnected with every hidden state of the neural network
## Supervised Learning:
|Input(x)|Output(y)|Application|
|--------|---------|-----------|
|Home Features|Price|Real Estate 
|Ad,user info|Click on add(0)|Online Advertising
|Image|Object(1...,1000)|Photo Tagging
|||

Types of data:
>Structured Data(Discrete and definitive , eg Properties of a house such as price size etc,)
>Unstructured Data(Less Definitive for example image ,Audio ,language)

Biggest Advantage of CNN's is with increase in Data the Performance of the Algorithm Increases more than that in traditional algirithms.
 
## Week 2:
## Deep Neural Networks

## Deep L-Layer Neural Networks

- Deep neural networks have multiple hidden layers.
- A deep network is more expressive and can learn complex patterns.

## Forward and Backward Propagation

- Forward propagation computes the output of the network for a given input.
- Backward propagation computes the gradients to update the network's parameters.

### Forward Propagation in Deep Networks

For \(l\) from 1 to L:
\[
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
\]
\[
A^{[l]} = g^{[l]}(Z^{[l]})
\]

Where:
- \(Z^{[l]}\) is the linear output of layer \(l\).
- \(A^{[l]}\) is the activated output of layer \(l\).
- \(W^{[l]}\) is the weight matrix for layer \(l\).
- \(b^{[l]}\) is the bias vector for layer \(l\).
- \(g^{[l]}\) is the activation function for layer \(l\).

### Backward Propagation in Deep Networks

Backward propagation computes gradients starting from the final layer and working backward.

For each layer \(l\):
\[
dZ^{[l]} = A^{[l]} - Y
\]
\[
dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]T}
\]
\[
db^{[l]} = \frac{1}{m}\sum_{i=1}^{m}dZ^{[l](i)}
\]
\[
dA^{[l-1]} = W^{[l]T}dZ^{[l]}
\]

Where:
- \(dZ^{[l]}\) is the gradient of the cost with respect to \(Z^{[l]}\).
- \(dW^{[l]}\) is the gradient of the cost with respect to \(W^{[l]}\).
- \(db^{[l]}\) is the gradient of the cost with respect to \(b^{[l]}\).
- \(dA^{[l-1]}\) is the gradient of the cost with respect to \(A^{[l-1]}\).
- \(m\) is the number of training examples.
- \(Y\) is the actual output.

## Hyperparameters

- Hyperparameters are parameters that are not learned from the data but are set manually.
- Examples include the learning rate (\(\alpha\)) and the number of hidden layers.

## Building Blocks of Deep Networks

- Deep networks consist of various building blocks, including fully connected layers and activation functions.

### Activation Functions

Common activation functions include:
- Sigmoid
- Tanh (hyperbolic tangent)
- ReLU (Rectified Linear Unit)
- Leaky ReLU

## Random Initialization

- Weights should be initialized randomly to break symmetry.
- Initializing all weights to zero would lead to symmetry, and all neurons would learn the same thing.

## Summary

- Week 2 introduced deep neural networks with multiple hidden layers.
- Forward and backward propagation were discussed.
- Hyperparameters and activation functions were explained.
- Random initialization of weights is important to avoid symmetry.

### Key Concepts
- Deep neural networks
- Forward and backward propagation
- Hyperparameters
- Activation functions (sigmoid, tanh, ReLU)
- Random initialization
  
## Week 3: 
## Shallow Neural Networks

## Neural Networks Overview

- Neural networks can have multiple layers, making them deep.
- Deep neural networks have become increasingly popular due to their ability to learn complex patterns.

## Shallow Neural Networks

- A shallow neural network typically consists of an input layer, one hidden layer, and an output layer.

### Activation Functions

- Activation functions introduce non-linearity to the neural network, allowing it to learn complex functions.
- Common activation functions include sigmoid, tanh, and ReLU.

### Forward Propagation

- Forward propagation is the process of computing the output of a neural network for a given input.

### Mathematical Representation of Forward Propagation

For a single hidden layer, forward propagation can be represented as:

\[
Z^{[1]} = W^{[1]}A^{[0]} + b^{[1]}
\]
\[
A^{[1]} = g^{[1]}(Z^{[1]})
\]
\[
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}
\]
\[
A^{[2]} = g^{[2]}(Z^{[2]})
\]

Where:
- \(Z^{[l]}\) is the linear output of layer \(l\).
- \(A^{[l]}\) is the activated output of layer \(l\).
- \(W^{[l]}\) is the weight matrix for layer \(l\).
- \(b^{[l]}\) is the bias vector for layer \(l\).
- \(g^{[l]}\) is the activation function for layer \(l\).

### Cost Function

- The cost function measures the dissimilarity between the predicted output and the actual output.
- Common cost functions include mean squared error (MSE) for regression and cross-entropy for classification.

### Backpropagation

- Backpropagation is used to compute the gradients of the cost function with respect to the network's parameters.
- Gradients are used to update the parameters using gradient descent.

## Vectorization

- Vectorization allows you to perform operations on entire datasets without explicit for loops.
- It significantly speeds up computations in deep learning.

## Summary

- Week 3 introduced shallow neural networks with one hidden layer.
- Activation functions introduce non-linearity.
- Forward propagation computes the network's output.
- Backpropagation computes gradients for parameter updates.
- Vectorization accelerates computations in deep learning.

### Key Concepts
- Shallow neural networks
- Activation functions (sigmoid, tanh, ReLU)
- Forward propagation
- Mathematical representation of forward propagation
- Cost functions (MSE, cross-entropy)
- Backpropagation
- Vectorization

## Week 4: 

## Practical Aspects of Deep Learning

- Deep learning has many practical considerations to make it work effectively.

## Regularization

- Regularization techniques help prevent overfitting and improve the generalization of deep neural networks.

### L2 Regularization (Weight Decay)

- L2 regularization adds a penalty term to the cost function that discourages large weight values.

\[
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^{L} ||W^{[l]}||_F^2
\]

Where \(\lambda\) is the regularization parameter.

### Dropout Regularization

- Dropout randomly drops neurons during training to prevent reliance on any single neuron.

## Gradient Checking

- Gradient checking is a technique to verify the correctness of backpropagation.

## Optimization Algorithms

- Optimization algorithms are used to update the network's parameters during training.

### Mini-Batch Gradient Descent

- Mini-batch gradient descent divides the training set into smaller batches for faster convergence.

### Momentum

- Momentum helps accelerate gradient descent by adding a fraction of the previous velocity to the current update.

### Adam Optimization Algorithm

- Adam combines ideas from RMSprop and momentum for efficient optimization.

## Learning Rate Decay

- Learning rate decay reduces the learning rate over time to fine-tune training.

## Batch Normalization

- Batch normalization normalizes the inputs of each layer to improve training speed and stability.

## Tips for Hyperparameter Tuning

- Hyperparameter tuning is essential for optimizing deep learning models.

## Summary

- Week 4 covered practical aspects of deep learning.
- Regularization techniques like L2 and dropout help prevent overfitting.
- Gradient checking verifies the correctness of backpropagation.
- Optimization algorithms like mini-batch gradient descent, momentum, and Adam are used for training.
- Learning rate decay and batch normalization improve training efficiency.

### Key Concepts
- Regularization (L2, dropout)
- Gradient checking
- Optimization algorithms (mini-batch gradient descent, momentum, Adam)
- Learning rate decay
- Batch normalization
- Hyperparameter tuning   
  
 