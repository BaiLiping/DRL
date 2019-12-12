import sys
#input from the user
s=input('print your name:')
print("hello, {}".format(s))

#tell the program to go to main if there is one

def main():
    print("printing from main")

if __name__=="__main__":
    main()
#main is not called by default, this is mostly used to break the intepreting order of the things


#interger division and floating point division
x=3
y=8
print("3//8 is {:.55f}".format(x//y)) #the bracket is a placeholder, inside is the formating info
print("3/8 is {:.55f}".format(x/y))

#defail print end with /n but can override it
print("helloworld1", end="")
print("helloworld2")

#UNICODE ENCODING
for i in range(65,65+26):
    print("{} is {}".format(chr(i),i))

#argument from command line and exit
if len(sys.argv)<2:
    print("missing commandline argument")
    exit(1) #exit(1) backto the shell #exit(0) 0 for success, other than 0 being failur:e
else:
    for i in range(len(sys.argv)):
        print(sys.argv[i])


