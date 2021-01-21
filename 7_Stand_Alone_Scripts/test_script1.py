import argparse

##This sets up a parser, which can read inputs given at the command line
parser = argparse.ArgumentParser(description='A test script to learn about argparse')

##Here we add an argument called --message, and add a description of how
##to use this argument via the 'help' keyword
parser.add_argument('--message', help='Enter a message to be printed. If your \
                    message contains spaces, you must wrap your message in \
                    quotation marks like this: --message="how fun"')

##This grabs all of the arguments out of the parser. Now all the arguments
##are attributes of args, which can access and use
args = parser.parse_args()

##message has become an attribute of args, so we can use it as we would
##any other variable. If the user hasn't input --message on the command line,
##args.message=None, so we can use it in a boolean test below
if args.message:
    print('Your message is: {:s}'.format(args.message))
else:
    print("You must enter some text via --message otherwise what's the point of me")
