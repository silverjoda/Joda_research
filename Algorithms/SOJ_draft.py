
import sys

def readnum():
    return int(sys.stdin.readline())

def readline():
    return raw_input()

def isprime(x):
    for i in range(2,x):
        if x % i == 0:
            return True
    return False

# Amount of test cases
n = readnum()

for i in range(n):

    # Getline:
    line = readline()

    a,b = line.split()

    # Do the stuff
    for j in range(a,b):
        if isprime(j):
            print j

    # Skip line
    print

