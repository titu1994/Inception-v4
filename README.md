# Inception v4 in Keras
Implementations of the Inception-v4, Inception - Resnet-v1 and v2 Architectures in Keras using the Functional API. The paper on these architectures is available at <a href="http://arxiv.org/pdf/1602.07261v1.pdf"><b>"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"</b></a>. 

The models are plotted and shown in the architecture sub folder. Due to lack of suitable training data (ILSVR 2015 dataset) and limited GPU processing power, the weights are not provided. 

## Inception v4
The python script '<b>inception_v4.py</b>' contains the methods necessary to create the Inception v4 network. 

Usage:
```
from inception_v4 import create_inception_v4

model = create_inception_v4()

```

## Inception ResNet v1
The python script '<b>inception_resnet_v1.py</b>' contains the methods necessary to create the Inception ResNet v1 network. 
It is to be noted that scaling of the residuals is turned <b>OFF</b> by default. This can be rectified by supplying 'scale=True' in the create method.

Usage:
```
from inception_resnet_v1 import create_inception_resnet_v1

model = create_inception_resnet_v1()

```

## Inception ResNet v2
The python script '<b>inception_resnet_v2.py</b>' contains the methods necessary to create the Inception ResNet v2 network. 
It is to be noted that scaling of the residuals is turned <b>ON</b> by default. 

There are a few differences in the v2 network from the original paper:<br>
<b>[1]</b> In the B blocks: 'ir_conv' nb of filters  is given as 1154 in the paper, however input size is 1152.<br>
    This causes inconsistencies in the merge-sum mode, therefore the 'ir_conv' filter size
    is reduced to 1152 to match input size.
    <br>
<b>[2]</b> In the C blocks: 'ir_conv' nb of filter is given as 2048 in the paper, however input size is 2144.<br>
    This causes inconsistencies in the merge-sum mode, therefore the 'ir_conv' filter size
    is increased to 2144 to match input size.
    
Usage:
```
from inception_resnet_v2 import create_inception_resnet_v2

model = create_inception_resnet_v2(scale=True)
```

# Architectures
## Inception v4

<img src="https://github.com/titu1994/Inception-v4/blob/master/Architectures/Inception-v4.png?raw=true">

## Inception ResNet v1

<img src="https://github.com/titu1994/Inception-v4/blob/master/Architectures/Inception%20ResNet-v1.png?raw=true">

## Inception ResNet v2

<img src="https://github.com/titu1994/Inception-v4/blob/master/Architectures/Inception%20ResNet-v2.png?raw=true">
