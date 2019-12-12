import pickle

a_dict={'a':1,'b':2}
wfile=open('pickleexample.pickle','wb')
pickle.dump(a_dict,wfile)
wfile.close()

rfile=open('pickleexample.pickle','rb')
r_dict=pickle.load(rfile)
rfile.close()

print(r_dict)

