import numpy as np
from IPython.core.debugger import Tracer

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  total_ele = x.size
  N = x.shape[0]
  X = np.reshape(x,(N,total_ele/N))
    
  # affine forward pass
  out = np.dot(X,w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # dout corresponds to the incoming delta
  # eq.3
  db = np.sum(dout,axis=0)

  # eq.4
  dw = np.dot(np.reshape(x,(x.shape[0],x.size/x.shape[0])).T, dout)
    
  # gradient wrt x, (assuming dx is eq.2 thus corresponds to a delta)
  dx = np.reshape(np.dot(dout,w.T),x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # only computes max(0,z)
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # eq.2: incorporates derivative of activation function
  dx = dout
  dx[x<=0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # compute mean of features
    batch_mean = np.mean(x,axis=0)
    
    # compute variance of features
    batch_var = np.var(x,axis=0)
    
    # compute running mean and variance
    running_mean = momentum * running_mean + (1 - momentum) * batch_mean
    running_var = momentum * running_var + (1 - momentum) * batch_var
    
    # normalise features
    x_norm = (x - batch_mean) / (np.sqrt(batch_var + eps))
    
    # scale and shift
    out = (gamma * x_norm) + beta
    cache = (x, x_norm, gamma, batch_mean, batch_var, eps)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    # compute output for test (inference) batch
    out = gamma / (np.sqrt(running_var + eps)) * x + (beta - gamma * running_mean / \
                                                      np.sqrt(running_var + eps))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  # assign useful parameters
  x, x_norm, gamma, batch_mean, batch_var, eps = cache
  x_mean = (x - batch_mean)
  std = np.sqrt(batch_var + eps)
  N = x.shape[0]

  # derivative wrt to beta
  dbeta = np.sum(dout,axis=0)

  # derivative wrt to gamma
  dgamma = np.sum(dout * x_norm, axis=0)
  
  # intermediate derivatives using a computational graph; refer fig. 2.1
  dxnorm = dout * gamma

  # eq.1
  imdt_11 = np.sum(dxnorm*x_mean, axis=0)
  imdt_12 = dxnorm * (1./std)
    
  # eq.2
  imdt_2 = imdt_11 * (-1./std**2)
    
  # eq.3
  imdt_3 = imdt_2 * (1./std) * 0.5
    
  # eq.4
  imdt_4 = imdt_3 * np.ones(x.shape) * (1./N)
    
  # eq.5
  imdt_5 = imdt_4 * x_mean * 2
    
  # eq.6
  imdt_61 = - np.sum(imdt_12 + imdt_5, axis=0)
  imdt_62 = imdt_12 + imdt_5
    
  # eq.7
  imdt_7 = imdt_61 * np.ones(x.shape) * (1./N)
    
  # eq.8
  dx = imdt_7 + imdt_62

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  # assign useful parameters
  x, x_norm, gamma, batch_mean, batch_var, eps = cache
  x_mean = x - batch_mean
  std = np.sqrt(batch_var + eps)
  N = x.shape[0]

  # derivative wrt to beta
  dbeta = np.sum(dout,axis=0)

  # derivative wrt to gamma
  dgamma = np.sum(dout * x_norm, axis=0)
  
  # intermediate derivatives
  #Tracer()()
  dxnorm = dout*gamma
  dvar = np.sum(dxnorm*x_mean, axis=0) * (-0.5) * 1./std**(3)
  dmean = np.sum(dxnorm, axis=0)*(-1./std) + dvar*np.sum(x_mean, axis=0)*(-2./N)
  
  # derivative wrt x
  dx = dxnorm * (1./std) + dvar*x_mean*(2./N) + dmean*(1./N)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p) / p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    mask = None
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  stride = conv_param['stride']
  pad = conv_param['pad']
    
  # define dimensions
  N = x.shape[0]
  C, H, W = x.shape[1:]
  HH, WW = w.shape[2:]
  F = w.shape[0]

  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # pad the input
  npad = ((0,0),(0,0),(pad,pad),(pad,pad))
  x_pad = np.pad(x,pad_width=npad,mode='constant',constant_values=0)
  
  # dimensions of output
  out_H, out_W = [(H + 2*pad - HH) / stride + 1, (W + 2*pad - WW) / stride + 1]
  out = np.zeros((N,F,out_H,out_W))

  # check validity of filter operation
  op_valid = ((H + 2*pad - HH) % stride == 0) and ((W + 2*pad - WW) % stride == 0)
  if(not op_valid):
    print "Error! Dimensions/parameters for spatial convolution incorrect"
  else:
    r = HH//2
    for row in xrange(out_H):
        c = WW//2
        for col in xrange(out_W):
            # define a window for convolution
            windowed_x = x_pad[:,:,r-HH//2:r+(HH+1)//2,c-WW//2:c+(WW+1)//2]
            
            # perform convolution
            out[:,:,row,col] = np.einsum('nijk,fijk->nf', windowed_x, w) + b

            # increment wrt stride
            c += stride

        # increment wrt stride
        r += stride
        
    """Alternate implementation with explicit loops for examples and filters
    
    for n in xrange(N):
        for f in xrange(F):
            r = HH//2
            for row in xrange(out_H):
                c = WW//2
                for col in xrange(out_W):
                    # define a window for convolution
                    windowed_x = x_pad[n,:,r-HH//2:r+HH//2,c-WW//2:c+WW//2]

                    # perform convolution
                    # out[n,f,row,col] = np.einsum('nijk,ijk->n', windowed_x, w[f,:,:,:]) + b[f]
                    out[n,f,row,col] = np.sum(windowed_x * w[f,:,:,:]) + b[f]

                    # increment wrt stride
                    c += stride

                # increment wrt stride
                r += stride"""
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  x,w,b,conv_param = cache  
    
  # define dimensions
  N = x.shape[0]
  C,H,W = x.shape[1:]
  F = w.shape[0]
  HH,WW = w.shape[2:]
  pad = conv_param['pad']
  stride = conv_param['stride']
  
  # matrix dimensions
  npad = ((0,0),(0,0),(pad,pad),(pad,pad))
  x_pad = np.pad(x,npad,mode='constant',constant_values=0)
  dx = np.zeros(x_pad.shape)
  db = np.zeros(F)
  dw = np.zeros(w.shape)

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  for n in xrange(N):
    for f in xrange(F):
        r = HH//2
        for row in xrange(dout.shape[2]):
            c = WW//2
            for col in xrange(dout.shape[3]):
                db[f] += dout[n,f,row,col]
                dw[f,:,:,:] += x_pad[n,:,r-HH//2:r+(HH+1)//2,c-WW//2:c+(WW+1)//2]*dout[n,f,row,col]
                dx[n,:,r-HH//2:r+(HH+1)//2,c-WW//2:c+(WW+1)//2] += w[f,:,:,:]*dout[n,f,row,col]

                # increment c
                c += stride

            # increment r
            r += stride

  # remove padding from dx
  dx = dx[:,:,1:H+1,1:W+1]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # define parameters
  pool_H = pool_param['pool_height']
  pool_W = pool_param['pool_width']
  stride = pool_param['stride']
    
  # define dimensions
  N,C,H,W = x.shape
  out_H, out_W = [(H-pool_H)/stride + 1, (W-pool_W)/stride + 1]
  out = np.zeros((N,C,out_H,out_W))
  
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  r = pool_H//2
  for row in xrange(out_H):
      c = pool_W//2
      for col in xrange(out_W):
          windowed_x = x[:,:,r-pool_H//2:r+(pool_H+1)//2,c-pool_W//2:c+(pool_W+1)//2]
          out[:,:,row,col] = np.amax(windowed_x, axis=(2,3))
           
          # increment c
          c += stride
          
      # increment r
      r += stride 
                
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  
  # unwrap cache
  x, pool_param = cache
  
  # define parameters
  pool_H = pool_param['pool_height']
  pool_W = pool_param['pool_width']
  stride = pool_param['stride']
  
  # define dimensions
  N,C,H,W = x.shape
  dx = np.zeros_like(x)
  dout_H, dout_W = [dout.shape[2], dout.shape[3]]  
    
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  for n in xrange(N):
        for chnl in xrange(C):
            r = pool_H//2
            for row in xrange(dout_H):
                c = pool_W//2
                for col in xrange(dout_W):
                    #Tracer()()
                    temp = x[n,chnl,r-pool_H//2:r+(pool_H+1)//2,c-pool_W//2:c+(pool_W+1)//2]
                    mask = (temp == np.amax(temp))+0
                    dx[n,chnl,r-pool_H//2:r+(pool_H+1)//2,c-pool_W//2:c+(pool_W+1)//2] += dout[n,chnl,row,col]*mask

                    # increment c
                    c += stride
                    
                # increment r
                r += stride
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # define dimensions
  N,C,H,W = x.shape
  
  mode = bn_param['mode']
    
  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  # reshape x
  x_res = np.reshape(x,(N*H*W,C))

  # compute batchnorm
  out, cache = batchnorm_forward(x_res, gamma, beta, bn_param)
  out = np.reshape(out,(N,C,H,W))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None
  
  # define parameters
  N,C,H,W = dout.shape  

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  # reshape dout
  dout_res = np.reshape(dout,(N*H*W,C))
  
  # backprop for spatial batchnorm
  dx, dgamma, dbeta = batchnorm_backward_alt(dout_res,cache)

  # reshape output values
  dx = np.reshape(dx,(N,C,H,W))
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx