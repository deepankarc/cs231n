import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]  
  num_classes = W.shape[1]
  scores = np.zeros((1,W.shape[1]))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for ex in xrange(num_train):
      # calculate the scores of each class
      scores = np.dot(X[ex,:],W)
      
      # to ensure numerical stability
      scores -= np.amax(scores)
      correct_y_score = scores[y[ex]]
      tot = np.sum(np.exp(scores))
        
      # cross-entropy loss
      #Tracer()()
      sfmax_prob = np.exp(correct_y_score)/tot
      loss += - np.log(sfmax_prob)
    
      # compute gradient
      for j in xrange(num_classes):
        if(j == y[ex]):
            dW[:,j] += - X[ex,:].T/(1+(np.exp(correct_y_score)/(tot-np.exp(correct_y_score))))
        else:
            dW[:,j] += np.exp(scores[j])/tot*X[ex,:].T
            
  # normalised loss wrt no. of examples
  loss /= num_train
  dW /= num_train
    
  # regularisation
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  # to ensure numerical stability
  scores -= np.amax(scores,axis=1,keepdims=True)
  scores = np.exp(scores)
    
  # compute correct y score probabilities
  correct_y_scores = (scores[np.arange(num_train),y])
  tot_scores = np.sum(scores,axis=1)
  
  # calculate cross-entropy loss
  loss = np.sum(-np.log(correct_y_scores/tot_scores))
    
  # compute gradient; refer fig. softmax
  prob_mat = scores/tot_scores[:,None]
  prob_mat[np.arange(num_train),y] -= 1
  dW = np.dot(X.T,prob_mat)
    
  # average across examples
  loss /= num_train
  dW /= num_train

  # regularisation
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

