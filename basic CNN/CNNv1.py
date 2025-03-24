import numpy as np

#x is the standard vector we use for each layer

def ReLU(x):
    return np.softmax(0,x)

def softmax(x):
    exps = np.exp(x-np.max(x)) #subtract max(x) to help with computations
    return exps

class FullyConnetedLayer:
    def __init__(self, input_d, output_d):

        self.W = np.random.randn(output_d, input_d) * np.sqrt(2. /input_d) #He initialisation
        self.b - np.zeros(output_d, 1)
    
    
    def forward(self, x):
        self.x = x  
        return np.dot(self.W, x) + self.b
    
    def backward(self, dout, learning_rate):

        dW = np.dot(dout, self.x.T)
        db = np.sum(dout, axis=1, keepdims=True)
        dx = np.dot(self.W.T, dout)
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dx

def cross_entropy_loss(predictions, labels):
    # predictions are output from softmax, labels are one-hot encoded

    loss = -np.sum(labels * np.log(predictions + 1e-8)) #not sure why we're subtracting 1e-8, I'm assuming it has something to do with yihat in the formula for this kind of loss
    return loss 


    #need to add convolution and pooling layers, implemented in a similar pattern


class ConvolutionLayer:
    def __init__(self, kernels, strides, padding, input_d, output_d, bias):
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.input_d = input_d
        self.output_d = output_d
        self.feature_maps = np.array(len(self.kernels), output_d)
    def forward(self, x):
        for i in range(len(self.kernels)):
            current_map = np.zeros(self.output_d[i])
            for j in range(0,self.output_d[i], self.strides[i]):
                pass



class PoolingLayer:
    pass