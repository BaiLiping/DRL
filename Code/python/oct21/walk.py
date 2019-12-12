'''
python method walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up

syntax
os.walk(top,topdown=True,oneero=None,followlinks=False)
generate the file names in a directory tree by walking the tree wither top down or bottom-up. For each directory in the tree rooted at the directory top, it yeilds a 3 tuple(dirpath, dirnames, filenames)

dirpath is a string,the path to the directory. 
dirnames is a list of the names of the subdirectories in dirpath including '.' and '..'

filenames is a list of the names of the non-directory files in dirpath. Note that the names in the lists contain no path components. To get a full path to a file or directory in dirpath, do os.path.join





'''

