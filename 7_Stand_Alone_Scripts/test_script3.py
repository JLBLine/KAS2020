
import sys
from astropy.cosmology import WMAP9 as cosmo

def cosmo_values_at_redshift(redshift):
    """Given a redshift, print out some interesting cosmological values
    from the WMAP9 astropy model"""

    print("Age of universe is",cosmo.age(redshift))
    print("The luminosity distance is",cosmo.luminosity_distance(redshift))
    print("The contribution of photons to the critical density is", cosmo.Ogamma(redshift))
    print("The contribution of matter to the critical density is", cosmo.Om(redshift))
    return

if __name__ == '__main__':

    import argparse

    ##This sets up a parser, which can read inputs given at the command line
    parser = argparse.ArgumentParser(description='Another test script to learn about argparse and __name__ == __main__')

    ##This is a required positional argument
    parser.add_argument('redshift', type=float,
        help='Enter a redshift to investigate')

    ##This grabs all of the arguments out of the parser. Now all the arguments
    ##are attributes of args, which can access and use
    args = parser.parse_args()

    ##Check the redshift isn't negative
    if args.redshift < 0:
        print("You can't have a negative redshift, try a different value")

    ##Otherwise, do something
    else:
        cosmo_values_at_redshift(args.redshift)
