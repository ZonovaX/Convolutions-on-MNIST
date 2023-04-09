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

### CNN's with learning rate of 0.001:
The results show that increasing the stride leads to better training and better accuracy than using Avg Pooling. The two and three convolutional layer models (each layer has stride 2) perform similarly, despite the 3 layer model having having 15% fewer parameters. However, using Max Pooling achieves much higher accuracy, even with this small learning rate.<br>
<br>
Models:<br>
The green line is with 2 conv layers with stride 1 and then AvgPool(4) into FC, ~6.5k param <br>
The light blue line is with 2 conv layers with stride 1 and then MaxPool(4) into FC, ~6.5k param <br>
The orange line is with 2 conv layers with stride 2 and no pooling into FC, ~6.5k param<br>
The grey line is with 3 conv layers with stride 2 and no pooling into FC, ~5.5k param<br>
![image](https://user-images.githubusercontent.com/12636792/230748239-261e3f41-866a-4a35-84e2-33dd674ff233.png)
![image](https://user-images.githubusercontent.com/12636792/230748221-8aab21f6-be65-421a-9bd6-9205f509a45d.png)



### CNN's with learning rate of 0.05:
Using this larger learning rate led to much better trained models across the board. Interestingly it led to the model with Avg Pooling actually becoming more effective than the models with larger stride, ultimately being equally as effective as Max Pooling. However, Max Pooling trained faster.  <br>
Models: <br>
The light blue line is with 3 conv layers with stride 2 and no pooling into FC ~ 5.5k param <br>
The orange line is with 2 conv layers with stride 2 and no pooling into FC ~ 6.5k param <br>
The dark blue line is with 2 conv layers with stride 1 and then AvgPool(4) into an FC ~ 6.5k param <br>
The grey line is with 2 conv layers with stride 1 and then MaxPool(4) into an FC ~ 6.5k param <br>
![image](https://user-images.githubusercontent.com/12636792/230748350-67b53db9-4756-4fcb-a31e-7428fdc97d67.png)
![image](https://user-images.githubusercontent.com/12636792/230748341-1d112a84-954a-46a1-8039-709856bf43b4.png)

