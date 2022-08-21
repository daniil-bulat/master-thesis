##############################################################################
##############################################################################
#                                                                            #
#                       Parameterization Functions                           #
#                               Master Thesis                                #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import numpy as np
import scipy
from scipy.stats import norm, skewnorm, stats
import matplotlib.pyplot as plt



# Function for Normal Distribution
def normal(mean, std, num_rev, color="black"):
    x = np.linspace(mean-4*std, mean+4*std, num_rev)
    p = scipy.stats.norm.pdf(x, mean, std)
    z = plt.plot(x, p, color, linewidth=2)




# Function for Skew Normal Distribution
def skew_normal(mean, std, skew, num_rev, color= 'red'):
    x = np.linspace(mean-4*std, mean+4*std, num_rev)
    p = skewnorm.pdf(x,skew,loc=mean, scale=std)
    z = plt.plot(x, p, color, linewidth=2)
    




