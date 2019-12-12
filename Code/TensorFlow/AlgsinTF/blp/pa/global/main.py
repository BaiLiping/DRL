from testglobal import testglobal
import numpy as np

def main():
    global state
    state=np.zeros(10)
    test=testglobal(state)
    #change the state after it has been passed to class
    state[0]=9
    test.class_method()
    print(state)
    test.class_method()
    print(state)
    test.class_method()
    print(state)
    test.class_method()
    print(state)



if __name__=='__main__':
    main()
