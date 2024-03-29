import numpy as np
from IPython.core.debugger import Tracer


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  # score for time step t
  z = np.dot(x,Wx) + np.dot(prev_h,Wh) + b
  next_h = np.tanh(z)
  
  cache = (z, x, prev_h, Wx, Wh)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  z, x, prev_h, Wx, Wh = cache
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  # ref. fig.1
  # derivative of tanh = sech^2 = 1 - tanh^2
  # Note: I reckon it's better to use the above over 1/cosh^2
  imdt_val = dnext_h * (1 - np.tanh(z)**2)
    
  # eq.1
  dprev_h = np.dot(imdt_val, Wh.T)
    
  # eq.2
  dWx = np.dot(x.T, imdt_val)
    
  # eq.3
  dWh = np.dot(prev_h.T, imdt_val)
    
  # eq.4
  db = np.sum(imdt_val, axis=0)

  # eq.5
  dx = np.dot(imdt_val, Wx.T)
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  # declare return values
  h = None
  cache = {}
  
  # get essential dimensions
  N,T,D = x.shape
  H = b.shape[0]
  h = np.zeros((N,T,H))
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  # compute h for every time step
  for t in xrange(T):
    x_t = np.reshape(x[:,t,:],(N,D))
    
    if(t == 0):
        h[:,t,:], cache[t] = rnn_step_forward(x_t, h0, Wx, Wh, b)
    else:
        h[:,t,:], cache[t] = rnn_step_forward(x_t, np.reshape(h[:,t-1,:],(N,H)), Wx, Wh, b)
        
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  _, x, _, _, _ = cache[0]
    
  # get essential dimensions
  N,T,H = dh.shape
  D = x.shape[1]

  # initialise gradient matrices
  dx = np.zeros((N,T,D))
  dh0 = np.zeros((N,H))
  dh_prev = np.zeros((N,H))
  dWx = np.zeros((D,H))
  dWh = np.zeros((H,H))
  db = np.zeros(H)
    
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  for t in xrange(T-1,-1,-1):
    dh_next = np.reshape(dh[:,t,:],(N,H)) + dh_prev
        
    dx_t, dh_prev, dWx_t, dWh_t, db_t = rnn_step_backward(dh_next, cache[t])
    
    dx[:,t,:] += dx_t
    dh0 = dh_prev
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None  
    
  # define relevant parameters
  N,T = x.shape
  V,D = W.shape

  # initialise variables
  out = np.zeros((N,T,D))

  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  """# Implementation using loop
  for t in xrange(T):        
        out[:,t,:] = W[np.reshape(x[:,t],N), :]"""
        
  out = W[x, :]
  cache = x, W

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """   
  dW = None

  # initialise variables      
  x, W = cache
  dW = np.zeros_like(W)
  
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  np.add.at(dW,x,dout)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
    
  # define dimensions
  N,D = x.shape
  H = prev_h.shape[1]    
    
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  # activation score
  z = np.dot(x,Wx) + np.dot(prev_h,Wh) + b
    
  # forget-gate layer
  f = sigmoid(z[:,H:2*H])
    
  # input-gate layer
  i = sigmoid(z[:,0:H])
  g = np.tanh(z[:,3*H:])

  # output-gate layer
  o = sigmoid(z[:,2*H:3*H])
  
  # compute next cell state
  next_c = prev_c*f + i*g

  # compute next hidden state
  next_h = o*np.tanh(next_c)
    
  cache = x, prev_h, prev_c, next_c, Wx, Wh, f, i, g, o, z[:,0:H], z[:,H:2*H],z[:,2*H:3*H],z[:,3*H:]
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    
  # unroll cache
  x, h_prev, c_prev, c_t, Wx, Wh,f, i, g, o, ai,af,ag,ao = cache
    
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  # compute output gate and cell state's derivative
  do = dnext_h * np.tanh(c_t)
  dc_t = dnext_h * (1 - np.tanh(c_t)**2) * o + dnext_c

  # compute forget gate, input gate, candidate gate and previous cell state's derivatives
  df = c_prev * dc_t
  di = g * dc_t
  dg = i * dc_t
  dprev_c = f * dc_t
    
  # compute score intermediates for each gate
  dz_f = f * (1 - f) * df
  dz_i = i * (1 - i) * di
  dz_g = (1 - g**2) * dg
  dz_o = o * (1 - o) * do

  # compute derivatives wrt weights and biases
  dz = np.concatenate((dz_i, dz_f, dz_o, dz_g), axis=1)
  dWx = np.dot(x.T, dz)
  dWh = np.dot(h_prev.T, dz)
  db = np.sum(dz, axis=0)
    
  # compute dx and dprev_h
  dx = np.dot(dz, Wx.T)
  dprev_h = np.dot(dz, Wh.T)
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
    
  # initialise dimensions
  N,T,_ = x.shape
  H = h0.shape[1]

  # initialise variables
  h = np.zeros((N,T,H))
  cache = {}

  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  # intial cell state  
  c0 = np.zeros_like(h0)
    
  # forward pass for LSTM over time T
  for t in xrange(T):
    if(t == 0):
        h[:,t,:], _prev_c, cache[t] = lstm_step_forward(np.squeeze(x[:,t,:]), h0, c0, Wx, Wh, b)
    else:
        h[:,t,:], _prev_c, cache[t] = lstm_step_forward(np.squeeze(x[:,t,:]), np.squeeze(h[:,t-1,:]), _prev_c, Wx, Wh, b)
        
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  x = cache[0][0]
    
  # initialise dimensions
  N,T,H = dh.shape
  D = x.shape[1]

  # define and initialise variables
  dx = np.zeros((N,T,D))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros((4*H))

  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  # initial hidden and cell state's derivative
  dnext_c = np.zeros((N,H))
  dh_t = np.zeros((N,H))
    
  # backward pass for LSTM over time T
  for t in xrange(T-1,-1,-1):
    dh_curr = np.squeeze(dh[:,t,:]) + dh_t
    
    # calculate derivatives
    dx_t, dh_t, dnext_c, dWx_t, dWh_t, db_t = lstm_step_backward(dh_curr, dnext_c, cache[t])
    
    dx[:,t,:] += dx_t
    dh0 = dh_t
    dWx += dWx_t
    dWh += dWh_t
    db += db_t
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

def sigmoid(z):
    """
    Sigmoid function.
    
    Inputs:
    z - input scores
    
    Outputs:
    res - activation values
    """
    res = 1. / (1 + np.exp(-z))
    return res

