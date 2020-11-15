from numpy import arange
from numpy.random import normal
from numpy import sqrt,pi,exp,linspace
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
from lmfit import minimize, Parameters

def straight_line(x,m,c):
    '''Returns a straight line with gradient m and
    intercept c: y = m*x + c'''
    return m*x + c

##Assign some variables
m, c  = 2.4, 1.5

##Make a range of independent variables
##Inputting a single number n gives a range
##of 0 to n-1
x_range = arange(50)

##Generate a straight_line
observed = straight_line(x_range,m,c)

##make some noise - set the number of points to be same
##as the length of x_range
nu,sigma = 0,3
noise = normal(nu,sigma,len(x_range))

## x += 1 is short for x = x + 1
##so the line below just adds the noise array to the observed array,
##and assigns the name observed to resulting array
observed += noise

##By adding 'bo' we asking to plot 'circles' instead of a line
##There are MANY plotting options for you to research.
plt.plot(x_range,observed,'o',label='Observed data')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.savefig('test_plots/random_data.png',bbox_inches='tight')

linear_model = LinearModel(prefix='line_')

linear_params = linear_model.make_params()
linear_params['line_slope'].set(value=1.0)
linear_params['line_intercept'].set(value=0.0)

linear_fit = linear_model.fit(observed, linear_params, x=x_range)

fit_values = linear_fit.eval_components(x=x_range)

slope = linear_fit.params['line_slope']
intercept = linear_fit.params['line_intercept']

print(slope)
print(slope.value,slope.stderr)



plt.plot(x_range,fit_values['line_'],'-',label='Linear fit')
plt.legend()
plt.savefig('test_plots/random_data_with_fit.png',bbox_inches='tight')
