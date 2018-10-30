"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        
        padding_size = int((kernel_size-1)/2) # kernel size is equal to filter size
        print("padding_size=",padding_size)
        #such padding creates the same size of conv layer
        # kernel or filter size is 7 -> 3 *7*7
        #input_size = input_dim[1] #(32 from (48000, 3, 32, 32))
        #output_size = (input_size+2*padding_size-kernel_size)/stride_conv + 1
        #channel_output = num_filters
        """
        self.conv1  = nn.Conv2d(channels, num_filters ,kernel_size, stride = stride_conv, padding = padding_size, bias = True) # 2d means that the result will be the 2 dim matrix

        self.pool= nn.MaxPool2d(pool, stride = stride_pool)
        #self.conv2  = nn.Conv2d(num_filters, 16 ,kernel_size, stride = stride_conv, padding = padding_size, bias = True) # 
        
        """
        print("h = ", height, "w = ", width)
        height_new = int((height  - pool) / stride_pool) + 1
        width_new =  int((width - pool) / stride_pool) + 1
        # if height = width 
        output_channel = int((height - pool) / stride_pool)+1 # we know we use same padding that means height in = height out after first conv1, width  in = width out
        print(output_channel , " - output channel", height_new, width_new)
        #self.fc1 = nn.Linear(output_channel * height_new * width_new, hidden_dim)
        
        
        
        self.conv1  = nn.Conv2d(channels, num_filters ,kernel_size, stride = stride_conv, padding = padding_size, bias = True)
        self.conv1.weight.data.mul_(weight_scale)
        self.pool = nn.MaxPool2d(pool, stride = stride_pool)
        self.fc1 = nn.Linear(num_filters*height_new*width_new, hidden_dim)
        self.dp1 =  nn.Dropout2d(p = dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        pass
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        
        #conv - relu - 2x2 max pool - fc - dropout - relu - fc
        # An efficient transition from spatial conv layers to flat 1D fully 
        # connected layers is achieved by only changing the "view" on the
        # underlying data and memory structure.
         #conv - relu - 2x2 max pool - VIEW - fc - dropout - relu - fc
        
        #conv - relu 
        """
        x = F.relu(self.conv1(x))
        print("conv1 - relu \n", x.size())
        # max pool 2
        #x = F.max_pool2d(x, 2)# _pair(pool) = (pool,pool) = (2,2)
        x = self.pool1(x) # is equivalent
        print("pool \n", x.size())#, y.size())
        #it is already pooling with stride 2
        ## view before fc
        
       # x = F.relu(self.conv2(x))
       # print("conv2 - relu \n", x.size())
        #
         
        print("num_of features ", self.num_flat_features(x))
        x = x.view(-1,self.num_flat_features(x))
        #x = x.view(-1,1568)
        print(" after view ", x.size())
        print("=================")
        #fc - dropout - relu - fc
        x = self.fc1(x)
        #x = self.dp1(x)
        x = F.relu(x)
        x = self.fc2(x)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.dp1(self.fc1(x)))
        x = self.fc2(x)
        pass
    
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x
    
    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        #print(x.size())
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
