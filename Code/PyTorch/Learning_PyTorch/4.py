'''
under the hood, each primitive autograd operator is really two functions that operator on tensors, the forward function computes output tensor from input tensor, the backward function receive the gradient of the output tensor with resepect to some scalar value and compute the gradient of the input tensor with respect to that same scalar value

in PyTorch, we can easily define our own autograd operator by defining a subclass of torch.augograd.Function and implementing the forward and backward functions. We can then use out new autograd operator by constructing an instance and calling it like any other function
'''
import torch

class MyReLU(torch.autograd.Function):
    '''
    any implemented autograd funciton is a subclass of torch.autograd.Function()
    '''
    @staticmethod
    def forward(ctx,input):
        '''
        in the forward pass we receive a tensor containing the input and return a tensor containing the output. ctx is a context object that can be used to stash information for backward computation. you need to use ctx.save_for_backward in order to pass this information along
        ctx is short for context
        '''
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx,grad_output):
        '''
        in the backward pass we receive a tensor containing the gradient of the loss with respect to the output, and we need to compute the gradient of the loss withrespect to the input
        '''
        input,=ctx.saved_tensors
        grad_input=grad_output.clone()
        grad_input[input<0]=0
        return grad_input

dtype=torch.float
device=torch.device('cpu')

N,D_in,H,D_out=64,1000,100,10

x=torch.randn(N,D_in,device=device,dtype=dtype)
y=torch.randn(N,D_out,device=device,dtype=dtype)

w1=torch.randn(D_in,H,device=device,dtype=dtype,requires_grad=True)
w2=torch.randn(H,D_out,device=device,dtype=dtype,requires_grad=True)

learning_rate=1e-6
for episode in range(500):
    relu=MyReLU.apply
    y_pred=relu(x.mm(w1).mm(w2))
    loss=(y_pred-y).pow(2).sum()
    print(episode,loss.item())
    loss.backward()
    with torch.no_grad():
        w1-=learning_rate*w1.grad
        w2-=learning_rate*w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
