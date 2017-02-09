import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvolutionalNetTest(object):
    """
    Describes a convolutional network with architecture:
    [conv - (spatialbn) - ReLU - conv - (spatialbn) - ReLU - pool] x M - [affine - (batchnorm)] x N - [softmax or SVM]
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    
    def __init__(self, input_dim=(3,32,32), num_filters=32, filter_size=3,
                conv_layers=2, affine_layers=2, hidden_dims=[100, 100], num_classes=10, 
                weight_scale=1e-3, reg=0.0, dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: List of number of units to use in the fully-connected hidden layers
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
    
        self.params = {}
        self.bn_params = {}
        self.spatialbn_params = {}
        self.use_bn = use_batchnorm
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.conv_layers = conv_layers
        self.affine_layers = affine_layers

        # define dimensions
        C,H,W = input_dim
        F = filter_size
        pad = (F-1)//2
        pool_S = 2
        H2,W2 = H,W

        # initialise weights, biases and batchnorm parameters for convolutional layers
        for l in xrange(2*conv_layers):
            self.params['W%d' % (l+1)] = np.random.randn(num_filters,C,F,F) * weight_scale
            self.params['b%d' % (l+1)] = np.zeros(num_filters)
            C = num_filters
            if(l % 2 == 1 and l != 0):
                H2,W2 = [(H2+2*pad-F)//pool_S + 1, (W2+2*pad-F)//pool_S + 1]
            if(self.use_bn is True):
                # initialise gamma and beta
                self.params['gamma%d' % (l+1)] = np.ones(C)
                self.params['beta%d' % (l+1)] = np.zeros(C)
                # initialise the running means and variances for spatial batchnorm
                self.spatialbn_params['running_mean'] = np.zeros(num_filters)
                self.spatialbn_params['running_var'] = np.zeros(num_filters)
      
        # initialise weights, biases and batchnorm parameters for affine layers
        if(affine_layers == 1):
            hidden_dims = [H2*W2*num_filters] + [hidden_dims] + [num_classes]
        else:
            hidden_dims = [H2*W2*num_filters] + hidden_dims + [num_classes]
            
        for l in xrange(affine_layers+1):
            self.params['W%d' % (l+2*conv_layers+1)] = np.random.randn(hidden_dims[l],hidden_dims[l+1]) * weight_scale
            self.params['b%d' % (l+2*conv_layers+1)] = np.zeros(hidden_dims[l+1])
            if(self.use_bn is True and l != affine_layers):
                # initial gamma and beta
                self.params['gamma%d' % (l+2*conv_layers+1)] = np.ones(hidden_dims[l+1])
                self.params['beta%d' % (l+2*conv_layers+1)] = np.zeros(hidden_dims[l+1])
                # initialise the running means and variances for spatial batchnorm
                self.bn_params['running_mean'] = np.zeros(hidden_dims[l+1])
                self.bn_params['running_var'] = np.zeros(hidden_dims[l+1])

        # convert parameter's datatype to float
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        
    
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
    
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # parameters for storing values
        conv_cache = {}
        fc_cache = {}
        scores = None
        N,num_F,H,W = [0,0,0,0]
        conv_a = X
        conv_layers = self.conv_layers
        affine_layers = self.affine_layers
        fp_check = False
        use_bn = self.use_bn
        spatialbn_params = self.spatialbn_params
        bn_params = self.bn_params
        
        # spatial batchnorm parameters
        use_bn = self.use_bn
        if(use_bn == True):
            if(y is None):
                bn_params['mode'] = 'test'
                spatialbn_params['mode'] = 'test'
            else:
                bn_params['mode'] = 'train'
                spatialbn_params['mode'] = 'train'
        
        # perform forward pass
        total_layers = conv_layers+affine_layers+1
        for l in xrange(total_layers):
            # forward pass for convolutional network
            if(l < conv_layers):
                # check if batchnormalisation is to be used
                if(use_bn is False):
                    conv_a, conv_cache[2*l+1] = conv_relu_forward(conv_a, self.params['W%d' % (2*l+1)], 
                                                                          self.params['b%d' % (2*l+1)], conv_param)
                    conv_a, conv_cache[2*l+2] = conv_relu_pool_forward(conv_a, self.params['W%d' % (2*l+2)], 
                                                                          self.params['b%d' % (2*l+2)], conv_param, pool_param)
                else:
                    conv_a, conv_cache[2*l+1] = conv_spatialbn_relu_forward(conv_a, self.params['W%d' % (2*l+1)], 
                                                                            self.params['b%d' % (2*l+1)], self.params['gamma%d' % (2*l+1)], 
                                                                            self.params['beta%d' % (2*l+1)], conv_param, spatialbn_params)
                    conv_a, conv_cache[2*l+2] = conv_spatialbn_relu_pool_forward(conv_a, self.params['W%d' % (2*l+2)], 
                                                                            self.params['b%d' % (2*l+2)], self.params['gamma%d' % (2*l+2)],
                                                                            self.params['beta%d' % (2*l+2)], conv_param, spatialbn_params,
                                                                            pool_param)
                    
                if(l == conv_layers-1):
                    fp_check = True
                    
            # forward pass for affine network
            else:
                if(fp_check is True):
                    fp_check = False
                    N,num_F,H,W = conv_a.shape
                    fc_a = np.reshape(conv_a,(N,num_F*H*W))
                
                # evaluate if batchnorm is to be used
                if(use_bn == False):
                    if(l == total_layers-1):
                        scores, fc_cache[l] = affine_forward(fc_a, self.params['W%d' % (2*conv_layers+l-1)], 
                                                             self.params['b%d' % (2*conv_layers+l-1)])
                    else:
                        fc_a, fc_cache[l] = affine_forward(fc_a, self.params['W%d' % (2*conv_layers+l-1)], 
                                                           self.params['b%d' % (2*conv_layers+l-1)])
                else:
                    if(l == total_layers-1):
                        scores, fc_cache[l] = affine_forward(fc_a, self.params['W%d' % (2*conv_layers+l-1)], 
                                                                       self.params['b%d' % (2*conv_layers+l-1)])
                    else:

                        fc_a, fc_cache[l] = affine_batchnorm_forward(fc_a, self.params['W%d' % (2*conv_layers+l-1)], 
                                                                     self.params['b%d' % (2*conv_layers+l-1)], 
                                                                     self.params['gamma%d' % (2*conv_layers+l-1)], 
                                                                     self.params['beta%d' % (2*conv_layers+l-1)], bn_params)
                    
        # for inference
        if y is None:
            return scores
        
        loss, grads = 0, {}
        
        # calculate loss and final layer delta_L
        loss, delta_l = softmax_loss(scores,y)
        
        # perform backpropagation
        if(use_bn == False):
            for l in xrange(total_layers-1,-1,-1):
                # perform backprop for affine layers
                if(l >= conv_layers):                
                    delta_l, grads['W%d' % (2*conv_layers+l-1)], grads['b%d' % (2*conv_layers+l-1)] = affine_backward(delta_l, fc_cache[l])
                    if(l == conv_layers):
                        delta_l = np.reshape(delta_l, (N,num_F,H,W))

                # perform backprop for convolutional layers
                else:
                    delta_l, grads['W%d' % (2*l+2)], grads['b%d' % (2*l+2)] = conv_relu_pool_backward(delta_l, conv_cache[2*l+2])
                    delta_l, grads['W%d' % (2*l+1)], grads['b%d' % (2*l+1)] = conv_relu_backward(delta_l, conv_cache[2*l+1])
        
        else:
            for l in xrange(total_layers-1,-1,-1):
                # perform backprop for affine layers
                if(l >= conv_layers and l == total_layers-1):                
                    delta_l, grads['W%d' % (2*conv_layers+l-1)], grads['b%d' % (2*conv_layers+l-1)] = affine_backward(delta_l, fc_cache[l])
                elif(l >= conv_layers):
                    delta_l, grads['W%d' % (2*conv_layers+l-1)], grads['b%d' % (2*conv_layers+l-1)], grads['gamma%d' % (2*conv_layers+l-1)], grads['beta%d' % (2*conv_layers+l-1)] = affine_batchnorm_backward(delta_l, fc_cache[l])

                    if(l == conv_layers):
                        delta_l = np.reshape(delta_l, (N,num_F,H,W))

                # perform backprop for convolutional layers
                else:
                    delta_l, grads['W%d' % (2*l+2)], grads['b%d' % (2*l+2)], grads['gamma%d' % (2*l+2)], grads['beta%d' % (2*l+2)] = conv_spatialbn_relu_pool_backward(delta_l, conv_cache[2*l+2])
                    delta_l, grads['W%d' % (2*l+1)], grads['b%d' % (2*l+1)], grads['gamma%d' % (2*l+1)], grads['beta%d' % (2*l+1)] = conv_spatialbn_relu_backward(delta_l, conv_cache[2*l+1])
        
        # add regularisation
        for l in xrange(total_layers):
            if(l < conv_layers):
                W1 = self.params['W%d' % (2*l+1)]
                W2 = self.params['W%d' % (2*l+2)]
                
                loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2))
                grads['W%d' % (2*l+1)] += self.reg*W1
                grads['W%d' % (2*l+2)] += self.reg*W2
            else:
                W = self.params['W%d' % (2*conv_layers+l-1)]
                
                loss += 0.5*self.reg*np.sum(W*W)
                grads['W%d' % (2*conv_layers+l-1)] += self.reg*W
            
        return loss, grads

pass