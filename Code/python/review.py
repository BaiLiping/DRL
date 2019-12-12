class test(object):
    def __init__(self,x):
        self.des='this is the init constructor'
        self.x=x
    def __call__(self,x):
        self.des='this is the call constructor'
        self.x=x
    def __str__(self):
        return 'test contains {} from'.format(self.x)+self.des


def main():
    test1=test(2)
    print(test(1))
    print(test(4))

    nums1=[ n*n for n in range(10)]
    print(nums1)

    #map(fun,seq)
    nums2=map(lambda n: n*n,range(10))
    print(next(nums2))
    print(next(nums2))
    print(next(nums2))
    print(next(nums2))

    #lambda is a nameless fucntion
    #it is possible to assign a name to a function

    func=lambda a,b: a*b
    print(func(4,5))

    def return_function(a):
        print(a)
        return return_function

    func=return_function(3)
    func(4)



if __name__=='__main__':
    main()
