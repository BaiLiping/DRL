import os
from datetime import datetime

#the os module allows one to interact with the environment, change environment variables, file path etc

#print out the current directory

directory=os.getcwd()
print(directory)

os.chdir('/home/blp/Desktop')

print(os.getcwd())

os.chdir(directory)
#you can pass a path to listdir(path) by default it would list the current working directory
os.mkdir('test')
os.mkdir('test/test1')
#the difference between mkdir and makedir is that the first can only create one level directory at one time while the other can create vested directories all at once
os.makedirs('test2/test3')
print(os.listdir())
os.rmdir('test/test1')
os.rmdir('test')
#the distinction between rmdir and removedirs are the same as mkdir and makedir
os.removedirs('test2/test3')
print(os.listdir())
os.mkdir('test')
os.rename('test','test_new')
print(os.listdir())

mod_time=os.stat('test_new').st_mtime
print(datetime.fromtimestamp(mod_time))

#generate a dirpath/dir/files
#for dirpath,dirnames,filenames in os.walk('/'):
    #print('Current Path:',dirpath)
    #print('Directories:',dirnames)
    #print('Files:',filenames)
    #print('\n')

print(os.environ.get('/home/blp'))
# you can just concatonate the string using functions provided by python
#file_path=os.environ.get('/home')+'test.txt'
#problem is that this method is error-prone, for instance you might forget the slashes etc

file_path=os.path.join('/home/blp/Desktop','test.txt')
print(file_path)
#can use os.path to split the filename from the directory name
#how to parsing filename and file manipulation
print(os.path.basename('/temp/test.txt'))
print(os.path.dirname('/temp/test.txt'))
print(os.path.split('/temp/test.txt'))
print(os.path.splitext('/temp/test.txt'))
#sometimes temp files are named without an extention


