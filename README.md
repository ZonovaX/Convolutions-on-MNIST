# Studying the Optimal Architecture of a Convolution Network
 
 Here we are studying the average learning rate and accuracy of convolutional networks with differing architectures, to develop a notion of "best practice." 
 
 ## Testing Methodology
 We test a variety of convolutional network architectures, measuring the loss (on training data) and the accuracy (on validation data) as training progresses. 
 
 
 We limit training to 15 epochs on the Fashion-Mnist data set(70k 28x28 images split into 10 categories), with other datasets to be tested in the future. 
 For each model, we average the loss and accuracy over 5 independent runs, using SGD with momentum = 0.9. 
 
 In addition, the weights are initialized using the orthogonal initialization scheme detailed in [Saxe - Exact solutions to the nonlinear dynamics of learning in
deep linear neural networks.](https://arxiv.org/pdf/1312.6120.pdf) This is done to reduce the variance in training due to the initial conditions compared to the typical Gaussian or Xavier initialization. 

## Results
### Comparing Pooling to Increased Stride for Reducing Parameters
We consider 2 convolution and 3 convolution models, with ReLU nonlinearity, connecting at the end to a final linear FC layer. FC layers are parameter intensive and scale with the size of the image being output by the convolutional layers. To reduce the number of parameters in the FC, one would like to reduce the image output by the convolutional layers. Two obvious methods of doing this are with pooling and by increasing the stride on the CNN's. Here we compare these two methods. 

Side note, models without final FC layers, which have the correct final dimensionality purely using convolutional layers with appropriately picked strides, performed significantly worse than models with final FC layers. 

#### CNN's with learning rate of 0.001:
The results show that increasing the stride leads to better training and better accuracy than using Avg Pooling. The two and three convolutional layer models (each layer has stride 2) perform similarly, despite the 3 layer model having having 15% fewer parameters. However, using Max Pooling achieves much higher accuracy, even with this small learning rate.<br>
<br>
Models:<br>
The green line is with 2 conv layers with stride 1 and then AvgPool(4) into FC, ~6.5k param <br>
The light blue line is with 2 conv layers with stride 1 and then MaxPool(4) into FC, ~6.5k param <br>
The orange line is with 2 conv layers with stride 2 and no pooling into FC, ~6.5k param<br>
The grey line is with 3 conv layers with stride 2 and no pooling into FC, ~5.5k param<br>
![image](https://user-images.githubusercontent.com/12636792/230748239-261e3f41-866a-4a35-84e2-33dd674ff233.png)
![image](https://user-images.githubusercontent.com/12636792/230748221-8aab21f6-be65-421a-9bd6-9205f509a45d.png)

#### CNN's with learning rate of 0.05:
Using this larger learning rate led to much better trained models across the board. Interestingly it led to the model with Avg Pooling actually becoming more effective than the models with larger stride, ultimately being equally as effective as Max Pooling. However, Max Pooling trained faster.  <br>
Models: <br>
The light blue line is with 3 conv layers with stride 2 and no pooling into FC ~ 5.5k param <br>
The orange line is with 2 conv layers with stride 2 and no pooling into FC ~ 6.5k param <br>
The dark blue line is with 2 conv layers with stride 1 and then AvgPool(4) into an FC ~ 6.5k param <br>
The grey line is with 2 conv layers with stride 1 and then MaxPool(4) into an FC ~ 6.5k param <br>
![image](https://user-images.githubusercontent.com/12636792/230748350-67b53db9-4756-4fcb-a31e-7428fdc97d67.png)
![image](https://user-images.githubusercontent.com/12636792/230748341-1d112a84-954a-46a1-8039-709856bf43b4.png)

## Parallel 3x3 and 5x5 Convolutional Layers vs Sequential Convolutional Layers
Naively, if one has two 3x3 convolutional layers in a row, you can capture patterns that are around 5x5 in size. Is there any benefit to having different kernal sizes within the same layer? Note, since the images here are only 28x28, there's very little room to increase the size of the kernal. The parallel convolutional models tried here are: <br>
(Pink line) A layer with parallel 3x3 and 5x5 convolutions with stride 2 and then MaxPool(3) into an FC ~ 5.5k param <br>
(Orange line) A 3x3 with stride 1 -> a 3x3 with stride 2 in parallel with a 5x5 with stride 2 -> a 3x3 with stride 1. These parallel parts each end with a maxpool(3) and then are connected to an FC ~ 6.6k param <br>

In the following plots, we compare these parallel models with the two previous 2-layer 3x3 stride 1 models which ended with AvgPool and MaxPool (Blue and Gray lines). We see that the parallel models do not achieve as good results as the sequential models
![image](https://user-images.githubusercontent.com/12636792/230802417-5638e2ca-166e-4b26-acfc-edd88c4ecc3f.png)
![image](https://user-images.githubusercontent.com/12636792/230802396-d55a8552-7404-42ab-b1b4-2a943c65c4c0.png)

## Inhibitive Nonlinearities, Inspired by GABA.
Neurons in the brain fire a variety of chemicals to control activation. As a simplistic model, we can classify these chemicals as either activation enhancing or inhibitting; GABA is an example of an inhibitor. Typical NN architectures add nonlinearity by using ReLU as the activation function, which activates for positive inputs, and outputs strictly positive values. In the brain, approximately 20% of neurons fire inhibitive chemicals (https://qbi.uq.edu.au/brain-basics/brain/brain-physiology/how-do-neurons-work); to simulate this in a neural network, let's consider a "negative" ReLU. here are two ways one might implement this. A normal ReLU follows the rules ReLU(x) = {x<0: 0, x>0: x}. We can consider a negative ReLU nReLU(x) = {x<0: 0, x>0: -x} or nReLU(x) = {x<0: x, x>0: 0}. <br>

The base network consists of two convolutional layers with kernal size 3 and stride 1 + ReLU's, connecting to a FC layer of size 100 + ReLU into a FC of size 10. We modify this by changing the 20% of the ReLU's between the two FC layers into nReLU's. This naive change didn't lead to a discernable difference in the performance of the networks; all three netowkrs performed equally well. Future testing of inhibitive activation functions should clamp the weights themselves to be strictly positive or negative, and could be deeper in terms of FC layers. 
![image](https://user-images.githubusercontent.com/12636792/231199108-9c204f89-d959-40ae-b9b5-75d6605897d1.png)
![image](https://user-images.githubusercontent.com/12636792/231199037-246f4dcd-3eb2-4c4e-a28e-0b8e6331527b.png)

