from functools import reduce

primes=[2,3,5,7,11,13]
def product(*numbers):
    for i in numbers:
        print(i)

product(*primes)
product(primes)
