at its core, PyTorch prodives two main features:
an n-dimentional tensor, similar to numpy but can be run on GPU
automatic differentiation for building and training neural-networks
(the major difference between pytorch and tensorflow is that tensorflow has to build the graph first and then exercute on that graph)

PyTOrch autograd looks a lot like TensorFlow, in both frameworks we define a computational graph and use automatic differentiation to compute gradients, the biggest difference between the two is that Tensorflow computational graph are static whereas Pytorch use dynamic computational graph.

In Tensorflow, we define the computational graph once and then execute the same graph over and over again, possibly deeding different input data to the graph. In PyTorch, each **forward pass** defines a new computational graph.

Static graphs are nice because you can optimize the graph up front, for example, a framework might decide to fuse some graph opration for efficiency or to come up with a strategy for distributing the graph across many GPUs or many machines. If you are reusing the same graph over and over, then this potentially costly up-front optimization can be amortized as the same graph is rerun over and over. 

One aspect where static and dynamic graphs differ is control flow. For some models we might wish to perform different computation for each data point; for example a recurrent newwork might be unrolled for different numbers of time steps for each data points, this unrolling can be implemented as a loop. WIth a static graph, the loop consruct needs to be a part of the graph. FOr this reason, TensorFlow provides operators such as tf.scan for embedding loops into the graph. With dynamic graphs the situation is simpler, since we build graph on the fly and each example, we can use normal imperative flow control to perform computation that differs for each input/

PyTorch and Autograd

in the above examples, we had to manually implement both the forward and the backward passes of our neural network. Manually implementing the backward pass is not a a big deal for a ssmall two layer network, but might become an accounting and notation nightmare should there be more layers.

PyTorch is known for its automatic differentiation capacity to autmate the compuation of backward passes in the newral networks. The autograd package in Pytorch provides exactly this functionality. WHen using autograd, the foward pass of your network will be used to define a computational graph(which is the major difference between this and tensorflow). nodes in the graph will be tensors, and edges will be functions that produce out [ut tensros from input tensors. backprop through this graph sthen allows you to easily compute gradients

This sounds complicated, but it is actulaly pretty simple. each tensor represents a notde in a computational graph. if x is a tensor that has x.requires_grad=True then x.grad is another tensor holding the gradient of x with respect to some scalar value.

(basically if you specify x.requires_grad=True, then you have access to x.grad, which can be used to update x, x-=x.grad)

under the hood, each primitive autograd operator is really two functions that operate on tensors, the forward function computes output tensor from input tensor, the backward function receive the gradient of the output tensor with respect to some scalar value and computes the gradient of the input tensor with respect to the same scalar value./

In PyTorch we can easily define out own autograd operator by defining a subclass of torch.autograd.Function and implement the forward and backward functions, we can then use our new autograd operator by constructing an instance and calling it like a cution passing tensor containing input data

need to have more understanding of python decorator



