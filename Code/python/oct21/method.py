class test(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return 3
    def __str__(self):
        return 'x={} y={}'.format(self.x,self.y)
    def __getitem__(self,index):
        if index==0:
            print(self.x)
        else:
            print(self.y)
        return None

def main():
    test_object=test(5,8)
    print(len(test_object))
    print(test_object)
    test_object[1]

if __name__=='__main__':
    main()
            
