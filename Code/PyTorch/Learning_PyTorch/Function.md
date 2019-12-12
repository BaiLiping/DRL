class torch.autograd.Function
records operation history and defines formulas for differentiation operations. Every operation performed on Tensor S creates a new function operation that performs the computation, and records that it happend. The history is retained in the form of a DAG of functions. with edges denoting data dependency.Then when backward is called, the graph is processed in the topoligical ordering, by calling backeard methods of each function object, and passing returned gradient on to the next function on this graph.

class Example(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
         result=i.exp()
         ctx.save_for_backward(result)
         return result
    @staticmethod
    def backward(ctx,grad_output):
        result, =ctx.saved_tensors
        return grad_output*result


it must accept a contex ctx as the first argument, followed by as many outputs did forward() return and it should return as many tensors as there were input to forwrad() each argument is the gradient wrt the given output and each returned value should be the gradient wo thte corresponding input 

the context can be used to retrieve tensor saved during the forward pass. it also has an atribute ctx.meed_input_grad as a tuple of booleans represeingitn whether each input needs gradient. 
