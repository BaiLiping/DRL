import logging
logging.basicConfig(filename='closurelog.log',level=logging.INFO)

def logger(func):
    def log_func(*args):
        logging.info('running {} with arguments {}'.format(func.__name__,args))
        #print(func(*args))
    return log_func

def add(x,y):
    return x+y

add_logger=logger(add)

add_logger(5,7)
add_logger(6,8)

