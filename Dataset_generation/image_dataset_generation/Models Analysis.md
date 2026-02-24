## Models Analysis:

# 1. ResNet-18:
- Specific type of CNN architecture designed for image classification tasks
- 18 layers deep with skip connections to mitigate vanishing gradient problem --> having more layers makes the gradients become tiny during backpropagation, making deep learning layers difficult to train.
- ~11 million parameters

Normal CNN Block:
Output = F(Input) where F is the function learned by the convolutional layers.

ResNet Block:
Output = F(Input) + Input

- Skip connections allow the model to learn residual differences (corrections) rather than the entire mapping, so the function to be learned is simpler and easier to optimize, enabling the training of deeper networks without degradation in performance.

Architecture:
- Initial convolutional layer with 7x7 kernel and stride of 2, followed by batch normalization, ReLU activation and max pooling.
- 4 residual stages, each containing 2 residual blocks:
    - Stage 1: 64 filters, output size 56x56
    - Stage 2: 128 filters, output size 28x28
    - Stage 3: 256 filters, output size 14x14
    - Stage 4: 512 filters, output size 7x7
The structure of the residual block is:
    - 3x3 Conv (stride 2 if downsampling, otherwise 1) --> BatchNorm --> ReLU --> 3x3 Conv (stride 1) --> BatchNorm --> Add input (skip connection) --> ReLU
- Final layers composed by global average pooling that gets the mean spatial value for each feature map (H x W x C --> 1 x 1 x C) and a fully connected layer that maps the features to the number of classes for classification.