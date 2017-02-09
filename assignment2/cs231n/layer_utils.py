from cs231n.layers import *
from cs231n.fast_layers import *

# affine - ReLU forward/backward pass
def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


# affine - PReLU forward/backward pass
def affine_prelu_forward(x, w, b, c):
  """
  Convenience layer that perorms an affine transform followed by a PReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - c: learnable parameter

  Returns a tuple of:
  - out: Output from the PReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, prelu_cache = prelu_forward(a, c)
  cache = (fc_cache, prelu_cache)
  return out, cache


def affine_prelu_backward(dout, cache):
  """
  Backward pass for the affine-prelu convenience layer
  """
  fc_cache, prelu_cache = cache
  da, dc = prelu_backward(dout, prelu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dc


# affine - batchnorm forward/backward pass
def affine_batchnorm_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """    
  #Tracer()()
  a, fc_cache = affine_forward(x, w, b)
  out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  cache = (fc_cache, bn_cache)
  return out, cache


def affine_batchnorm_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache = cache
  da, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta


# conv - ReLU forward/backward pass
def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


# conv - PReLU forward/backward pass
def conv_prelu_forward(x, w, b, c, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - c: learnable parameter in prelu
  
  Returns a tuple of:
  - out: Output from the PReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, prelu_cache = prelu_forward(a, c)
  cache = (conv_cache, prelu_cache)
  return out, cache


def conv_prelu_backward(dout, cache):
  """
  Backward pass for the conv-prelu convenience layer.
  """
  conv_cache, prelu_cache = cache
  da, dc = prelu_backward(dout, prelu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dc


# conv - ReLU - pool forward/backward pass
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


# conv - PReLU - pool forward/backward pass
def conv_prelu_pool_forward(x, w, b, c, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, prelu_cache = prelu_forward(a, c)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, prelu_cache, pool_cache)
  return out, cache


def conv_prelu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, prelu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da, dc = prelu_backward(ds, prelu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dc


# conv - spatial batchnorm - ReLU forward/backward pass
def conv_spatialbn_relu_forward(x,w,b,gamma,beta,conv_param,sbn_param):
  """
  A convenience layer that performs a convolution and spatial batchnormalisation 
  followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - bn_param: Parameters for batchnorm layer
  - pool_param: Parameters for the pooling layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  a, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, sbn_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, sbn_cache, relu_cache)
  return out, cache


def conv_spatialbn_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, sbn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(da, sbn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta


# conv - spatial batchnorm - ReLU - pool forward/backward pass
def conv_spatialbn_relu_pool_forward(x, w,b , gamma, beta, conv_param, sbn_param, pool_param):
  """
  A convenience layer that performs a convolution and spatial batchnormalisation 
  followed by a ReLU and max-pooling.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - bn_params: Parameters for batchnorm layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  a, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, sbn_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, sbn_cache, relu_cache, pool_cache)
  return out, cache  


def conv_spatialbn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, sbn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(da, sbn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta


# ----------------- utility layers for debugging ----------------- #
# ----------------- ---------------------------- ----------------- #
def conv_spatialbn_relu_pool_forward_test(x, w,b , gamma, beta, conv_param, sbn_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  #Tracer()()
  a, conv_cache = conv_forward_naive(x, w, b, conv_param)
  a, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, sbn_param)
  a, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, sbn_cache, relu_cache, pool_cache)
  return out, cache