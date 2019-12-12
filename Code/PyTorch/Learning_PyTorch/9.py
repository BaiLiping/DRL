'''
as an example of dynamic graphs and weight sharin, we implement a very strange mode: a fullying connected ReLU networ that on each forward pass chooses a random number between 1 and 4 and use that many hidden layers resuing the same weights multiple times to compute the innermost hidden layers

for this model, we can use normal python flow control to implement the loop and we can implement weight sharing amoung the innermost layers by simply usig the same mdoule multiple times when defining the forward pass
'''

import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super().__init__()
        self.input_linear=torch.nn.Linear(D_in,H)
        self.middle_linear=torch.nn.Linear(H,H)
        self.output_linear=torch.nn.Linear(H,D_out)
    def forward(self,x):
        h_relu=self.input_linear(x).clamp(min=0)
        for i in range(random.randint(0,3)):
            h_relu=self.middle_linear(h_relu).clamp(min=0)
        y_pred=self.output_linear(h_relu)
        return y_pred
N,D_in, H, D_out=64,1000,100,10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)
model=DynamicNet(D_in,H,D_out)
loss_fn=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
for episode in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(episode,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


