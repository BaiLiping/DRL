'''
decorator is related to first-class function, closure

first-class function, a programming language is said to have first class funcitons if it treats functions as first-class citizens

a first class citizen(first class object) in a programming language is an entity which supports all the operatins generally availabel to other entieties. These operations typically include being passed as an argument, return from a cunction and assinged to a variable
'''

def html_tag(tag):
    def wrap_text(message):
        print('<{0}>{1}</{0}>'.format(tag,message))
    return wrap_text

print_h1=html_tag('h1')
print_h1('test')
print_h1('this is the message')

print_h2=html_tag('h2')
print_h2('message')
