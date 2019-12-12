def outer_function_1():
    message="hi from outter function 1"
    def inner_function():
        print(message)
    return inner_function()

def outer_function_2():
    message='hi from outter function 2'
    def inner_function():
        print(message)
    return inner_function #without the parathesis means you return the function itself without exercuting it.
#the benefit of this kind of structure is for passing of variables

def outer_function_3(message):
    def inner_function():
        print(message)
    return inner_function
#you can imagine you have to pass the variable once, and then this structure remembers that

#decorator does exactly what the closure does but much easier syntax
def decorator_function(original_function):
    def wrapper_function(*args,**kwargs):
        print('additional functionality added by the wrapper function')
        return original_function(*args,**kwargs)
    return wrapper_function

def display():
    print('this is the display function')

decorated_display=decorator_function(display)
decorated_display()

@decorator_function
def display_info(name,age):
    print('name {} age {}'.format(name,age))
display_info('john',25)


#the reason peoplel use decorator functions is to add functionality to existing functions
#a easier syntax is 
@decorator_function
def display_function():
    print('I am displaying')

display_function()

class decorator_class(object):
    def __init__(self,original_function):
        self.original_function=original_function

    def __call__(self,*args,**kwargs):
        print('call method exercuted before {}'.format(self.original_function.__name__))
        return self.original_function(*args,**kwargs)

@decorator_class
def display_info(name,age):
    print('name {} age {}'.format(name,age))
display_info('john', 25)


outer_function_1()
f=outer_function_2()
f()
f2=outer_function_3('this is function 3')
f2()




