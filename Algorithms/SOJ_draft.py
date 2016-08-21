
import sys

def readnum():
    return int(sys.stdin.readline())

def readline():
    return raw_input()


# Amount of test cases
n = readnum()

for i in range(n):

    # Getline:
    line = readline()

    # Length of the line
    length = len(line)

    # String which we will iterate over:
    str = line[:length/2]

    # Do the stuff
    for j,c in enumerate(str):

        # skip every second char
        if j % 2 != 0:
            continue

        # Print char without newline
        sys.stdout.write(c)

    # Skip line
    print

