"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.
Notes:
The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.
# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os
from numpy.core.umath_tests import inner1d


class Activation(object):

    """
    Interface for activation functions (non-linearities).
    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        self.state = 1.0 / (1.0 + np.exp(-x))

        return self.state

    def derivative(self):

        # Maybe something we need later in here...
        last_result = self.state

        return last_result * (1.0 - last_result)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        m = (np.exp(x)) ** 2
        m = (m - 1.0) / (m + 1.0)
        self.state = m

        return m

    def derivative(self):
        return 1.0 - (self.state) ** 2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        x[ x < 0 ] = 0
        self.state = x
        return x

    def derivative(self):
        x = self.state
        x[ x > 0 ] = 1.0
        return x


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y
        

        a = np.max(x, 1).reshape(-1, 1)

        log_sum_exp = a + np.log(np.sum(np.exp(x-a), 1).reshape(-1, 1))

        log_yhat = x - log_sum_exp
        self.sm = log_yhat
        return -np.sum(y * log_yhat, 1)

    def derivative(self):

        # self.sm might be useful here...
        yhat = np.exp(self.sm)  # the actual output of NN(probability)

        return yhat - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.nbatch = 0

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        self.x = x
        momentum = 0.9

        if eval:
            B = np.shape(x)[0]
            mean = self.running_mean / self.nbatch
            var = self.running_var * B / (B - 1) / self.nbatch
            # var = self.running_var / self.nbatch
            # print(B, self.nbatch)
            self.norm = (x - mean) / np.sqrt(var + self.eps)
            self.out = self.gamma * self.norm + self.beta
            # scale = self.gamma / np.sqrt(var + self.eps)
            # self.out = x * scale + (self.beta - mean * scale)
        else:
            self.nbatch += 1
            self.mean = np.mean(x, 0)
            self.var = np.var(x, axis=0)  # np.mean((x - self.mean) ** 2, axis=0)

            self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta

            # update running batch statistics
            self.running_mean += self.mean
            self.running_var += self.var
            # self.running_mean = momentum * self.running_mean + (1 - momentum) * self.mean
            # self.running_var = momentum * self.running_var + (1 - momentum) * self.var

        # ...
        return self.out

    def backward(self, delta):
        # delta is the derivatives of error w.r.t. y
        # shape of delta: batch_size * output_size
        N = self.x.shape[0]

        dout_ = delta * self.gamma
        a = -0.5 * (self.var + self.eps) ** (-1.5)
        # dvar = a * np.sum(inner1d(dout_, (self.x - self.mean)))
        dvar = np.sum(dout_ * (self.x - self.mean) * a, axis=0)
        dx_ = 1 / np.sqrt(self.var + self.eps)

        # self.dgamma = np.sum(inner1d(delta, self.norm))

        # intermediate for convenient calculation
        dvar_ = 2 * (self.x - self.mean) / N
        di = dout_ * dx_ + dvar * dvar_
        dmean = -1 * np.sum(di, axis=0)
        dmean_ = np.ones_like(self.x) / N

        dx = di + dmean * dmean_
        self.dbeta = np.sum(delta, 0)
        self.dgamma = np.sum(delta * self.norm, axis=0)
        return dx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    # W = np.random.randn(ndim, ndim); w, s, v = np.linalg.svd(W)
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros((1, d))


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = [weight_init_fn(*t) for t in zip((input_size, *hiddens), (*hiddens, output_size))]
        self.dW = [np.zeros(t) for t in zip((input_size, *hiddens), (*hiddens, output_size))]
        self.b = [bias_init_fn(d) for d in (*hiddens, output_size)]
        self.db = [np.zeros((1, d)) for d in (*hiddens, output_size)]
        if momentum:
            self.mW = [np.zeros(t) for t in zip((input_size, *hiddens), (*hiddens, output_size))]
            self.mb = [np.zeros((1, d)) for d in (*hiddens, output_size)]
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            out_size = hiddens + [output_size]
            self.bn_layers = [BatchNorm(d) for d in out_size[:self.num_bn_layers]]
            # BatchNorm(hiddens[self.num_bn_layers - 1])

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        y = x

        for i in range(self.nlayers):
            z = y @ self.W[i] + self.b[i]
            if self.bn and i < self.num_bn_layers:
                z = self.bn_layers[i](z, not self.train_mode)

            y = self.activations[i](z)

        self.output = y  # shape: batch_size * output_size
        return y

    def zero_grads(self):
        for i in range(self.nlayers):
            self.dW[i].fill(0)
            self.db[i].fill(0)

    def step(self):
        if not self.momentum:
            for i in range(self.nlayers):
                self.W[i] -= self.lr * self.dW[i]
                self.b[i] -= self.lr * self.db[i]
                if self.bn and i < self.num_bn_layers:
                    bn = self.bn_layers[i]
                    bn.gamma -= self.lr * bn.dgamma
                    bn.beta -= self.lr * bn.dbeta

        else:
            self.momentum_update()

    def momentum_update(self):
        for i in range(self.nlayers):
            self.mW[i] = self.momentum * self.mW[i] - self.lr * self.dW[i]
            self.mb[i] = self.momentum * self.mb[i] - self.lr * self.db[i]
            self.W[i] += self.mW[i]
            self.b[i] += self.mb[i]

    def zero_monmentum(self):
        pass

    def backward(self, labels):
        loss = self.criterion(self.output, labels)
        if not self.train_mode:
            return np.sum(loss)

        partial_y = self.criterion.derivative()

        for i in range(self.nlayers-1, -1, -1):
            partial_z = self.activations[i].derivative()

            dz = partial_z * partial_y  # shape: batch_size * output_size
            if self.bn and i < self.num_bn_layers:
                dz = self.bn_layers[i].backward(dz)

            if i != 0:
                actfn_out = self.activations[i-1].state
            else:
                actfn_out = self.input
            a = actfn_out[..., np.newaxis]
            a = np.moveaxis(a, 0, -1)
            b = dz[..., np.newaxis]
            b = np.moveaxis(b, [0, -1], [-1, 0])
            self.dW[i] = np.mean(a * b, 2)  # matrix multiplication, but use broadcasting
            self.db[i] = np.mean(dz, 0)
            partial_y = dz @ self.W[i].T  # shape: batch_size * output_size

        return np.sum(loss)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...
        # shuffle the training set
        np.random.shuffle(idxs)
        trainx = trainx[idxs, :]
        trainy = trainy[idxs, :]

        train_loss = 0
        error_count = 0
        val_loss = 0

        mlp.train()
        for b in range(0, len(trainx), batch_size):
            # Train ...

            output = mlp(trainx[b:b + batch_size])
            y = trainy[b:b + batch_size]
            mlp.zero_grads()
            diff = np.argmax(output, 1) - np.argmax(y, 1)
            error_count += np.count_nonzero(diff)
            loss = mlp.backward(y)
            train_loss += loss
            if mlp.momentum:
                mlp.momentum_update()
            else:
                mlp.step()
        s = train_loss / len(trainx)
        z = error_count / len(trainx)
        training_losses.append(s)
        training_errors.append(z)
        print('epoch: {:5} loss: {:5.2f} error: {:5.2f}'.format(e, s, z))

        mlp.eval()
        error_count = 0
        for b in range(0, len(valx), batch_size):
            # Val ...
            output = mlp(valx[b:b + batch_size])
            y = valy[b:b + batch_size]
            diff = np.argmax(output, 1) - np.argmax(y, 1)
            error_count += np.count_nonzero(diff)
            val_loss += mlp.backward(y)
        validation_losses.append(val_loss / len(valx))
        validation_errors.append(error_count / len(valx))

        # Accumulate data...

    # Cleanup ...
    training_losses = np.array(training_losses)
    training_errors = np.array(training_errors)
    validation_losses = np.array(validation_losses)
    validation_errors = np.array(validation_errors)

    mlp.eval()
    error_count = 0
    loss = 0
    for b in range(0, len(testx), batch_size):
        output = mlp(testx[b:b + batch_size])
        y = testy[b:b + batch_size]
        diff = np.argmax(output, 1) - np.argmax(y, 1)
        error_count += np.count_nonzero(diff)
        loss += mlp.backward(y)

    print("test error rate: ", error_count / len(testx))
    print("test loss: ", loss / len(testx))

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

