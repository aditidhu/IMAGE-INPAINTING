# Course 4 :
## Week 1:
### Foundations of Convolutional Neural Networks

## Convolutional Neural Networks (CNNs)

- CNNs are a class of deep neural networks commonly used for image-related tasks.
- They excel at capturing spatial hierarchies and patterns.

## Edge Detection Example

- The process of edge detection illustrates how CNNs can learn features.

### Convolution Operation

- Convolution is a fundamental operation in CNNs.

#### Convolution of an Image with a Filter

- The mathematical representation of convolution:

\[
S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(m, n)K(i - m, j - n)
\]

Where:
- \(I\) is the input image.
- \(K\) is the filter (kernel).
- \(S\) is the output.

### Padding

- Padding is adding zeros around the input image to control the size of the output.

### Strided Convolution

- Strided convolution skips pixels in the output.

### Convolutions Over Volume

- Convolution can be applied to 3D volumes (RGB images or volumes in intermediate layers).

### One Layer of a Convolutional Network

- A single layer in a CNN consists of multiple filters.

### Simple Convolutional Network Example

- Example of a simple CNN architecture.

## Pooling Layer

- Pooling is used to reduce the size of the representation and computational cost.

### Max Pooling

- Max pooling selects the maximum value from a group of values.

## Understanding Convolutions

- Understanding the receptive field and how convolutional layers capture patterns.

## Summary

- Week 1 introduced the foundations of convolutional neural networks.
- CNNs are designed for image-related tasks and excel at feature learning.
- Convolutions, padding, and strided convolution are key concepts.
- Max pooling is used to reduce the size of representations.

### Key Concepts
- Convolutional Neural Networks (CNNs)
- Convolution operation
- Padding and strided convolution
- Pooling layer (max pooling)
- Receptive field


## Week 2: 
## Deep Convolutional Models: Case Studies

## Classic Networks

- Classic networks serve as the foundation for modern deep learning in computer vision.

### LeNet-5

- LeNet-5 is a pioneering convolutional neural network architecture.
- It was designed for handwritten digit recognition.

### AlexNet

- AlexNet was a breakthrough architecture that won the ImageNet competition.
- It introduced concepts like ReLU and dropout.

### VGG

- VGG networks are known for their simplicity and use of small filters (3x3).
- They are deep networks with many layers.

## Residual Networks (ResNets)

- ResNets introduced residual connections to address the vanishing gradient problem.

### Residual Block

- A residual block adds the input to the output, creating a "shortcut" connection.

### Identity Block

- An identity block has the same input and output dimensions.

### Convolutional Block

- A convolutional block changes the dimensions of the input.

### Why ResNets Work

- ResNets allow the gradient to flow more directly during training.

## Inception Network

- Inception networks use multiple filter sizes in parallel.

### Inception Block

- An inception block combines the outputs of different filter sizes.

## Transfer Learning

- Transfer learning leverages pre-trained models for new tasks.

### Fine-Tuning

- Fine-tuning adapts a pre-trained model to a new task by updating some layers.

## Data Augmentation

- Data augmentation artificially increases the size of the training set by applying transformations.

## Object Detection

- Object detection combines localization and classification.

### Sliding Windows

- Sliding windows is a classic method for object detection.

### Anchor Boxes

- Anchor boxes are used in modern object detection methods.

## Face Recognition

- Face recognition involves verifying or identifying faces.

### Siamese Network

- A Siamese network learns to distinguish between pairs of images.

## Neural Style Transfer

- Neural style transfer combines the content of one image with the style of another.

## Summary

- Week 2 covered deep convolutional models and case studies.
- Classic networks like LeNet-5, AlexNet, and VGG were introduced.
- Residual networks (ResNets) address the vanishing gradient problem.
- Inception networks use multiple filter sizes.
- Transfer learning, data augmentation, object detection, face recognition, and neural style transfer were discussed.

### Key Concepts
- LeNet-5
- AlexNet
- VGG
- Residual Networks (ResNets)
- Identity and convolutional blocks
- Inception networks
- Transfer learning
- Fine-tuning
- Data augmentation
- Object detection (sliding windows, anchor boxes)
- Face recognition (Siamese network)
- Neural style transfer

## Week 3: 


## Object Detection

- Object detection combines localization and classification.

### Localization vs. Classification

- Localization refers to identifying the location of an object within an image.
- Classification assigns a label to the object.

### Object Localization

- Object localization involves predicting the coordinates of the bounding box around an object.

## Landmark Detection

- Landmark detection identifies key points on objects, such as facial landmarks.

## Object Detection with Bounding Box

- Bounding boxes are used to locate objects within an image.

### Intersection over Union (IoU)

- IoU measures the overlap between predicted and ground truth bounding boxes.

\[
\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
\]

## Non-max Suppression

- Non-max suppression is used to eliminate redundant bounding box predictions.

## Anchor Boxes

- Anchor boxes are used to detect multiple object types of varying shapes and sizes.

## YOLO Algorithm

- YOLO (You Only Look Once) is an object detection algorithm that divides the image into a grid.

### YOLO Architecture

- YOLO predicts bounding boxes and class probabilities for each grid cell.

### YOLO Loss Function

- The YOLO loss function balances object localization and classification losses.

## Region Proposal

- Region proposal methods propose potential object regions for further processing.

### R-CNN

- R-CNN (Region-Based Convolutional Neural Network) is a classic object detection framework.

### Fast R-CNN

- Fast R-CNN improved the speed and efficiency of R-CNN.

### Faster R-CNN

- Faster R-CNN introduced the Region Proposal Network (RPN) for region proposals.

## Mask R-CNN

- Mask R-CNN extends Faster R-CNN to include instance segmentation.

## Summary

- Week 3 focused on object detection and segmentation.
- Object detection combines localization and classification.
- Techniques include bounding boxes, landmark detection, anchor boxes, YOLO, region proposal, and instance segmentation (Mask R-CNN).

### Key Concepts
- Object detection
- Localization vs. classification
- Bounding boxes
- Intersection over Union (IoU)
- Non-max suppression
- Anchor boxes
- YOLO algorithm
- Region proposal (R-CNN, Fast R-CNN, Faster R-CNN)
- Instance segmentation (Mask R-CNN)

## Week 4: 

## Special Applications: Face Recognition & Neural Style Transfer

## Face Recognition

- Face recognition involves verifying or identifying faces.

### One Shot Learning

- Traditional face recognition may not work well with a single training example.

### Siamese Network

- A Siamese network learns to distinguish between pairs of images.

### Triplet Loss

- Triplet loss is used to train Siamese networks.

\[
\mathcal{L}(A, P, N) = \max(\text{distance}(f(A), f(P)) - \text{distance}(f(A), f(N)) + \alpha, 0)
\]

Where:
- \(A\) is the anchor image.
- \(P\) is a positive (same person) image.
- \(N\) is a negative (different person) image.
- \(f(\cdot)\) represents the encoding of an image.
- \(\alpha\) is the margin.

## Neural Style Transfer

- Neural style transfer combines the content of one image with the style of another.

### Content Cost

- The content cost measures the similarity of features in content and generated images.

### Style Cost

- The style cost measures the similarity of Gram matrices between style and generated images.

### Total Cost

- The total cost combines content and style costs with hyperparameters.

## Face Verification & Binary Classification

- Face verification involves verifying whether two images belong to the same person.

### Siamese Network for Verification

- A Siamese network with a threshold is used for face verification.

## Neural Style Transfer: Art Generation

- Neural style transfer can generate artistic images by changing the style image.

### Gatys et al. Algorithm

- The Gatys et al. algorithm separates content and style for neural style transfer.

## Summary

- Week 4 covered special applications of deep learning.
- Face recognition with Siamese networks and triplet loss was discussed.
- Neural style transfer combines content and style from images.
- Face verification and artistic style generation were introduced.

### Key Concepts
- Face recognition
- One-shot learning
- Siamese network
- Triplet loss
- Neural style transfer
- Content and style cost
- Total cost
- Face verification
- Gatys et al. algorithm
