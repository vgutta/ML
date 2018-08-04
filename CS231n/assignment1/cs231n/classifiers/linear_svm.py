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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i] 
        dW[:, y[i]] -= X[i]
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W

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
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  #margins = np.maximum(0, scores - scores[y] + 1)
  #margins[y] = 0
    
  correct_label_score_idxes = (range(scores.shape[0]), y)
  # length of this vector is N, one correct label score per datapt
  correct_label_scores = scores[correct_label_score_idxes]

  # subtract correct scores (as a column vector) from every cell
  scores_diff = scores - np.reshape(correct_label_scores, (-1, 1))
  
  # add 1 for the margin.
  scores_diff += 1
  
  # now zero out all the loss scores for the correct classes.
  scores_diff[correct_label_score_idxes] = 0

  # now zero out all elements less than zero. (b/c of the max() in the hinge)
  indexes_of_neg_nums = np.nonzero(scores_diff < 0)

  scores_diff[indexes_of_neg_nums] = 0
    
  loss = scores_diff.sum()

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  pass
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
  scores_diff[scores_diff > 0] = 1
  correct_label_vals = scores_diff.sum(axis=1) * -1
  scores_diff[correct_label_score_idxes] = correct_label_vals

  dW = X.T.dot(scores_diff)
  dW /= num_train
  # add the regularization contribution to the gradient
  dW += reg * W
  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
