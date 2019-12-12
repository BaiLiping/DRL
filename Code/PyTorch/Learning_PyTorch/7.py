'''
up to this point we have updated the weights of our models by manually mutating the tensors holding the learnable parameters, this is not a huge burden for small graphs but in practice we often train neuralnet with sophisticated optimizers such as AdaGrad, RMSProp, Adam etc

the optim package in PyTorch abstrats the ide of an optimization algorithm and provide implementation of commonly used optimization algorithms
'''

import torch

N,D_in, H, D_out=64,1000,100,10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

model=torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out)
        )
loss_fn=torch.nn.MSELoss(reduction='sum')

learning_rate=1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for episode in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(episode,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


