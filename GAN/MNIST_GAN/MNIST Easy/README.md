#  MNIST GAN FROM SCRATCH
Using default or recommened parameters <br />
 <br />
Parameters: <br />
Image Size: 28X28X1 <br />
Batch Size = 100 <br />
Epoch = 200 <br />
## Generator: <br /> 
Input Dimension Noise: 100 <br />
Ouput Dimension 784 <br />
Layer 1 : 100 -> 256 <br />
Layer 2: 256 -> 512 <br />
Layer 3: 512 -> 1024 <br />
Layer 4: 1024 -> 784 <br />
TANH <br />
<br />
## Discriminator<br />
Input Dimension: 784 <br />
Layer 1: 784 -> 1024 <br />
Layer 2: 1024 -> 512  <br />
Layer 3: 512 -> 256 <br />
Layer 4: 256 -> 1 <br />
Sigmoid <br />
<br />
## Loss <br />
BCE Loss

<br />

## Result:
<img src="https://github.com/arpit2412/Generative-Adversarial-Network-/blob/master/GAN/MNIST_GAN/MNIST%20Easy/sample_epoch_200.png">
<br />

Reference 1: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py <br />
Reference 2: https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
