class BatchNorm(object):
    def __init__(self,in_dimmention,alpha):
        self.alpha=alpha
        self.epsilon=1e-8
        self.x=None
        self.norm=None
        self.out=None

        self.var=np.ones((1,in_dimmention))
        self.mean=np.zeros((1,in_dimmention))

        self.gamma=np.ones((1,in_dimmention))
        self.dgamma=np.zeros((1,in_dimmention))

        self.beta=np.zeros((1,in_dimmention))
        self.dbeta=np.zeros((1,in_dimmention))

        self.running_mean=np.zeros((1,in_dimmention))
        self.running_var=np.ones((1,in_dimmention))

    def __call__(self,x,eval=False):
        return self.forward(x,eval)

    def forward(self,x,eval=False):
        if eval:
            self.x=x
            self.mean=np.mean(x,axis=0)
            self.var=np.var(x,axis=0)
            self.norm=(x-self.mean)/np.sqrt(self.var=self.epsilon)
            self.out=self.out

    def backward(self,delta):
        
