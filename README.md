#  Attention Augmented CNN on CIFAR-100

## Overview

This project implements a **Simple Attention Augmented CNN** inspired by the paper *"Attention Augmented Convolutional Networks" (Bello et al., ICCV 2019)* and tests it on the **CIFAR-100** dataset. A simplified CNN is used with attention-convolution to validate the concept and test its effectiveness (instead of ResNet-50).

---

##  Dataset: CIFAR-100

* **60,000** color images (32x32) in **100 classes**.

  * **50,000** training images
  * **10,000** test images
* Data normalization used (to stableize training and faster convergence):

  ```python
  mean = (0.5071, 0.4865, 0.4409)
  std  = (0.2673, 0.2564, 0.2762)
  ```
 
---

##  Model Architecture

```plaintext
1-Input (3x32x32)

2-Conv2D (3→64) + (BatchNorm + ReLU) (standard design pattern in modern CNNs)

3-AugmentedConv2D (64→128) with Multi-Head Attention

4-AdaptiveAvgPool2D (1x1) (to collapse the spatial information to a global feature vector per image)

5-Fully Connected Layer (128→100) (it takes the 128 features from the pooled output and maps them to 100 output classes)

6-Softmax (CrossEntropyLoss)
```


###  AugmentedConv Details

* Combines standard convolution (for locality) with attention (for global context).
* Uses **multi-head relative attention**:

  * convolution filter size = **3x3**
  * Padding = **1** (to keep the spatial size of the input)
  * Stride = **1**
  * heads (`Nh`) = **4**
  * key dimensions (`dk`) = **40**
  * value dimensions (`dv`) = **40**
  * Output channels = **128**
* attention uses relative positional encoding for spatial reasoning.

---

##  Hyperparameters & Design Choices

| Hyperparameter  | Value | Rationale                                             |
| --------------- | ----- | ----------------------------------------------------- |
| Batch Size      | 128   | Common default for fast training and stable gradients |
| Epochs          | 10    | Enough for demonstration on CIFAR-100                 |
| Learning Rate   | 0.001 | Works well with Adam optimizer                        |
| Optimizer       | Adam  | Adaptive method, good for early convergence           |
| dk, dv          | 40    | Balanced key/value dimensionality without overfitting |
| Nh (heads)      | 4     | Small number of attention heads for CIFAR scale       |
| Output Channels | 128   | Doubles from input (64) to allow wider representation |

---

##  Performance


```
Test Accuracy on CIFAR-100: 27.58%
```
### Comparison with the Original Paper

| Model                                         | Dataset    | Accuracy |
|----------------------------------------------|------------|----------|
| **our Implementation (SimpleAugmentedCNN)** | CIFAR-100  | **27.58%** |
| **Attention Augmented Wide-ResNet-28-10 (paper)** | CIFAR-100  | **82.9%** |

###  Why Is our Accuracy Lower?

1. **Shallow Architecture**  
   our model is simple with only 1 augmented block and much fewer layers. The paper uses Wide-ResNet-28-10, which has 28 layers and 10× width so we have a much smaller parameter count. we couldn’t do the same as we don’t have strong computational power.

2. **Short Training time**  
   Our model was trained for only **10 epochs**, whereas the paper trains models for **100–200 epochs** .


3. **Learning Rate**  
   A fixed learning rate of 0.001 was used, while the paper uses advanced learning rate scheduling .


---

##  References

* Bello et al. (2019). [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925). ICCV 2019.
* CIFAR-100 Dataset: [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---
