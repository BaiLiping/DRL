class MLP(object):
    def __init__(self,input_size,output_size,hiddens,activations, weight_init_fn,bias_init_fn, criteerion, lr, momentum, num_bn_layers=0):
        self.train_mode=True
        self.num_bn_layers=num_bn_layers
        self.bn=num_bn_layers>0
        self.nlayers=len(hiddens)+1
        self.input_size=input_size
        self.output_size=output_size
        self.activations=activations
        self.criterion=criterion
        self.lr=lr
        self.momentum=momentum
        self.wights=[weight_init_fn(*t) for t in zip((input_size,*hiddens),(*hiddens,output_size))]
        self.dw=[np.zeros(t) for t in zip ((input_size,hiddens),(*hiddens,output_size))]
        self.b=[bias_init_fn(d) for d in (*hiddens,output_size)]
        self.db=[np.zeros((1,d)) for d in (*hiddens,output_size)]
        if momentum:
            self.mW=[np.zeros(t) for t in zip ((input_size,*hiddens),(*hiddens,output_size))]
            self.mb=[np.zeros((1,d)) for d in (*hiddens,output_size)]
            if self.bn:
                outsize=hiddens+[output_size]
                self.bn_layers=[BatchNorm(d) for d in outsize[:self.num_bn_layers]]
            self.input=None
            self.output=None
    def forward(self,x):
        self.input=x
        y=x
        for i in range(self.nlayers):
            z=y@self.weights[i]+self.b[i]
            if self.bn and i < self.num_bn_layers:
                z=self.bn_layers[i](z,not self.train_mode)
            y=self.activations[i](z)

        self.output=y
        return y
    def zero_grads(self):
        for i in range(self.nlayers):
            self.dW[i].fill(0)
            self.db[i].fill(0)
    def step(self):
        if not self.momentum:
            for i in range(self.nlayers):
                self.weights[i]-= self.lr* self.dw[i]
                self.b[i]-=self.lr*self.db[i]
                if self.db and i < self.num_bn_layers:
                    bn=self.bn_layers[i]
                    bn.gamma-=self.lr* bn.dgamma
                    bn.beta-=self.lr*bn.dbeta
        else:
            self.momentum_update()
    def momentum_update(self):
        for i in range(self.nlayers):
            self.mW[i]=self.momentum*self.mW[i]-self.lr*self.dW[i]
            self.mb[i]=self.momentum*self.mb[i]-self.lr*self.db[i]
            self.W[i]+=self.mW[i]
            self.b[i]+=self.mb[i]

    def zero_momentum(self):
        pass
    def backward(self,labels):
        loss=self.criterion(self.output,labels)
        if not self.train_mode:
            return np.sum(loss)
        partial_y=self.criterion.derivative()
        for i in range(self.nlayers-1,-1,-1):
            partial_z=self.activations[i].derivative()
            dz=partial_z*partial_y
            if self.bn and i<self.num_bn_layers:
                dz=self.bn_layers[i].backward(dz)
            if i != 0:
                actfn_out=self.activations[i-1].state
            else:
                actfn_out=self.input
            a=actfn_out[]


