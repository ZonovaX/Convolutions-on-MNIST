# Studying the Optimal Architecture of a Convolution Network
 
 Here we are studying the average learning rate and accuracy of convolutional networks with differing architectures, to develop a notion of "best practice." 
 
 ## Testing Methodology
 We test a variety of convolutional network architectures, measuring the loss (on training data) and the accuracy (on validation data) as training progresses. 
 
 We limit training to 15 epochs on the Fashion-Mnist data set(70k 28x28 images split into 10 categories), with other datasets to be tested in the future. 
 For each model, we average the loss and accuracy over 5 independent runs. 
 
 In addition, the weights are initialized using the orthogonal initialization scheme detailed in [Saxe - Exact solutions to the nonlinear dynamics of learning in
deep linear neural networks.](https://arxiv.org/pdf/1312.6120.pdf) This is done to reduce the variance in training due to the initial conditions compared to the typical Gaussian or Xavier initialization. 

