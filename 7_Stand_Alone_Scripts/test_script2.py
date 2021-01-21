import argparse
import sys
import numpy as np

##This sets up a parser, which can read inputs given at the command line
parser = argparse.ArgumentParser(description='Another test script to learn about argparse')

parser.add_argument('--integer', type=int,
    help='Enter an integer to square.')

parser.add_argument('--float', type=float,
    help='Enter a float to square.')

parser.add_argument('--also_do_sqrt', action='store_true',
    help='If added, will also perform a square root on the input number')

##This grabs all of the arguments out of the parser. Now all the arguments
##are attributes of args, which can access and use
args = parser.parse_args()

##If the user hasn't entered a number, there is nothing to do, so exit
if not args.integer and not args.float:
    ##Little advanced this - you could just print a warning here, but if
    ##another programme is relying on output from this script, it's easier
    ##to catch errors by actually properly exiting
    sys.exit("Please enter either one or both of --integer / --float, otherwise there is nothing for the script to do")

##Otherwise, do something
else:
    ##if the user has entered an integer, do something with it
    if args.integer:
        ##in both cases we will print the int squared, so calculate here
        sqrd = args.integer*args.integer
        ##If the user has asked, also do the square root
        if args.also_do_sqrt:
            sqrt = np.sqrt(args.integer)
            print('Your integer is {:d}, squared={:.3e}, sqrt={:.3e}'.format(args.integer,sqrd,sqrt))
        ##Otherwise, just plot out the sqrt
        else:
            print('Your integer is {:d}, squared={:.3e}'.format(args.integer,sqrd))

    if args.float:
        ##in both cases we will print the int squared, so calculate here
        sqrd = args.float*args.float
        ##If the user has asked, also do the square root
        if args.also_do_sqrt:
            sqrt = np.sqrt(args.float)
            print('Your float is {:.3e}, squared={:.3e}, sqrt={:.3e}'.format(args.float,sqrd,sqrt))
        ##Otherwise, just plot out the sqrt
        else:
            print('Your float is {:.3e}, squared={:.3e}'.format(args.float,sqrd))
