class constructor(object):
    def __init__(self):
        self.w=10
        self.b=11
        print('this is from init, w={}'.format(self.w))
    def __call__(self,x):
        
        return x*self.w+self.b
        #print('this is from call, y={}'.format(y))
'''
    def __str__(self):
        return 'w={}'.format(self.w)
'''
if __name__=='__main__':
    a=constructor() #this method call the init constructor
    #constructor(2)#this method call the call constructor, both constructors initialize the same attribute
    #print(constructor(3))
    b=a(4) #after you instanciated the object with constructor __init__ you can call the instance with __call__, for regular methods, a.methodname(), for __call__ you just go for a() as if a is a function itself, this is sort of, defining that instance as a function.
    print(b)


