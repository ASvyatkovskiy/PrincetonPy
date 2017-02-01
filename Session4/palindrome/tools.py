#!/usr/bin/env python

def digits(x):
    digs = []
    while x != 0:
        div,mod = divmod(x,10)
        digs.append(mod)
        x = mod
    return digs


def is_palindrome(x):
    digs = digits(x)
    for f,r in zip(digs, reversed(digs)):
        if f != r:
            return False
    return True	    
