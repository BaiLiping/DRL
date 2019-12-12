sentense1='this is sentense 1'
sentense2='this is sentense 2'
sentense3='this is sentense 3'

def print_sentenses(*args):
    for i in args:
        print(i)

def print_key_word(**kwargs):
    for key,value in kwargs.items():
        print(key,value)
        #print(value)


print_sentenses(sentense1,sentense2,sentense3)
print_key_word(sentens1='1',sentense2='2',sentense3='3')


