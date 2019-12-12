'''
sometimes you will want tot specify models that are more complex than a sequence of existing modules for these case, you can define your own modules by subclassing nn.Module and defining a forwrad which receives input tensor and produces output tensor using other modele or other autograd operations on tensor
'''

import torch
class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super().__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)

    def forward(self,x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred

N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

model=TwoLayerNet(D_in,H,D_out)
loss_fn=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)

for episode in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(episode,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


