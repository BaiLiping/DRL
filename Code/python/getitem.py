# __call__ defines function()
# __getitem__ defines function[]

class Foo:
    def __getitem__(self,key):
        print(key)
        return None

foo=Foo()
foo['abc']
foo[1]
