# Inception v4 in Keras
Implementations of the Inception-v4, Inception - Resnet-v1 and v2 Architectures in Keras using the Functional API. Paper on these architectures 
is available at <a href="http://arxiv.org/pdf/1602.07261v1.pdf"><b>"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"</b></a>.

## Inception v4
The python script '<b>inception_v4.py</b>' contains the methods necessary to create the Inception v4 network. 

## Inception ResNet v1
The python script '<b>inception_resnet_v1.py</b>' contains the methods necessary to create the Inception ResNet v1 network. 
It is to be noted that scaling of the residuals is turned <b>OFF</b> by default. This can be rectified by supplying 'scale=True' in the create method.

## Inception ResNet v2
The python script '<b>inception_resnet_v2.py</b>' contains the methods necessary to create the Inception ResNet v2 network. 
It is to be noted that scaling of the residuals is turned <b>OFF</b> by default. This can be rectified by supplying 'scale=True' in the create method.

There are a few differences in the v2 network from the original paper:<br>
<b>[1]</b> In the B blocks: 'ir_conv' nb of filters  is given as 1154, however input size is 1152.<br>
    This causes inconsistencies in the merge-add mode, therefore the 'ir_conv' filter size
    is reduced to 1152 to match input size.
    <br>
<b>[2]</b> In the C blocks: 'ir_conv' nb of filter is given as 2048, however input size is 2144.<br>
    This causes inconsistencies in the merge-add mode, therefore the 'ir_conv' filter size
    is increased to 2144 to match input size.
  
  Currently trying to find a proper solution with original nb of filters.
