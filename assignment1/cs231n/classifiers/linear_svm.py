import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      # does not add correct class score margin to the loss
      if (j == y[i]):
        continue
      
      # calculates the margins; note delta = 1
      margin = scores[j] - correct_class_score + 1
      d_count = 0
      if (margin > 0):
        # count for no. of additions of derivative of correct class
        d_count += 1
        loss += margin
        # for incorrect classes
        dW[:,j] += X[i,:].T
      
      # for correct classes
      dW[:,y[i]] -= d_count*X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)

  # retrieves correct class score from the calculated scores using y
  correct_y_score = np.array([[scores[ex, y[ex]] for ex in xrange(num_train)]])
  margins = scores - correct_y_score.T + 1 # delta = 1

  # sums all positive margin values
  loss = np.sum(margins[(margins > 0) & (margins != 1)])
    
  # calculate average loss
  loss /= num_train

  # add regularisation term
  loss += 0.5*reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # creates a mask which stores the constant to be multiplied to the derivative
  # for correct and incorrect classes
  mask = np.zeros(margins.shape)
  
  # incorrect classes have multiple 1
  mask[(margins > 0) & (margins != 1)] = 1
  
  # correct classes have a multiple defined by no. of positive margins
  d_count = np.sum(mask,axis=1)

  # create complete mask
  mask[np.arange(num_train),y] = -d_count
  
  # compute gradients
  dW = np.dot(X.T,mask)
    
  # adds regularisation term and averages over samples
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
