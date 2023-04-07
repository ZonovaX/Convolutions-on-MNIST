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

### CNN's with learning rate of 0.001:
The results show that increasing the stride leads to better training and better accuracy than using pooling. The two and three convolutional layer models (each layer has stride 2) perform similarly, despite the 3 layer model having having 15% fewer parameters. 

![image](https://user-images.githubusercontent.com/12636792/230689552-408d9c98-fd87-4016-8850-28519b208476.png)
