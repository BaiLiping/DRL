class Employee(object):
    def __init__(self,first,last,pay):
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@company.com'

    def fullname(self):
        return '{} {}'.format(self.first,self.last)

    def apply_raise(self):
        self.pay=int(self.pay*1.04)

class Developer(Employee):
    def __init__(self,first,last,pay,programming_language):
        super().__init__(first,last,pay)
        #Employee.__init__(self,first,last,pay)
        self.programming_language=programming_language

dev_1=Developer('Liping','Bai','100000','cpp')
#print(help(Developer))
print(dev_1.programming_language)
