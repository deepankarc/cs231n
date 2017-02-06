import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerPreluConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.bn_params = {}
    self.sbn_params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_bn = use_batchnorm
    
    # define dimensions
    C,H,W = input_dim
    F = filter_size
    pad = (F-1)//2
    pool_S = 2
    H2, W2 = [(H+2*pad-F)/pool_S + 1, (W+2*pad-F)/pool_S + 1]
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.randn(num_filters,C,F,F)*weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(H2*W2*num_filters, hidden_dim)*weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale
    self.params['b3'] = np.zeros(num_classes)
    self.params['c1'] = np.array([0.25])
    self.params['c2'] = np.array([0.25])
    
    if(self.use_bn == True):
        # initialise gamma and beta for batch norm
        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)
        
        # initialise running_mean and variance for batchnorm and spatial bn
        self.sbn_params['running_mean'] = np.zeros(num_filters)
        self.sbn_params['running_var'] = np.zeros(num_filters)
        self.bn_params['running_mean'] = np.zeros(hidden_dim)
        self.bn_params['running_var'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    c1, c2 = self.params['c1'], self.params['c2']
    
    # spatial batchnorm parameters
    use_bn = self.use_bn
    if(use_bn == True):
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        if(y is None):
            self.bn_params['mode'] = 'test'
            self.sbn_params['mode'] = 'test'
        else:
            self.bn_params['mode'] = 'train'
            self.sbn_params['mode'] = 'train'
        
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # perform convnet forward pass
    conv_a1, conv_cache = conv_prelu_pool_forward(X, W1, b1, c1, conv_param, pool_param)
    
    if(use_bn == True):
        conv_a1, sbn_cache = spatial_batchnorm_forward(conv_a1, gamma1, beta1, self.sbn_params)
    
    # define dimensions
    N,F,H,W = conv_a1.shape

    # perform FCnet forward pass
    a1_flattened = np.reshape(conv_a1,(N,F*H*W))
    # fc_a1, fc1_cache = affine_relu_forward(a1_flattened,W2,b2)
    fc_a1, fc1_affine_cache = affine_forward(a1_flattened,W2,b2)
    if(use_bn == True):
        fc_a1, bn_cache = batchnorm_forward(fc_a1, gamma2, beta2, self.bn_params)
    fc_a1, fc1_prelu_cache = prelu_forward(fc_a1, c2)
    scores, fc2_cache = affine_forward(fc_a1,W3,b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # compute loss and output layer's grad_W
    loss, delta_L = softmax_loss(scores,y)
    
    # backprop for FCnet
    delta_l2, grads['W3'], grads['b3'] = affine_backward(delta_L, fc2_cache)
    delta_l1, grads['c2'] = prelu_backward(delta_l2, fc1_prelu_cache)
    if(use_bn == True):
        delta_l1, grads['gamma2'], grads['beta2'] = batchnorm_backward(delta_l1, bn_cache)
    delta_l1, grads['W2'], grads['b2'] = affine_backward(delta_l1, fc1_affine_cache)
    # delta_l1, grads['W2'], grads['b2'] = affine_relu_backward(delta_l2, fc1_cache)
    
    # reshape delta_l1 for convnet
    delta_l1_res = np.reshape(delta_l1,(N,F,H,W))
    
    # backprop for spatial batchnorm
    if(use_bn == True):
        delta_l1_res, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(delta_l1_res, sbn_cache)
    
    # backprop for convnet
    _, grads['W1'], grads['b1'], grads['c1'] = conv_prelu_pool_backward(delta_l1_res, conv_cache)
    
    # add regularisation
    loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    grads['W1'] += self.reg*W1
    grads['W2'] += self.reg*W2
    grads['W3'] += self.reg*W3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
