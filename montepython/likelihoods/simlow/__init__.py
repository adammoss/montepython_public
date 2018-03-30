import os
from montepython.likelihood_class import Likelihood_prior


class simlow(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        tau = cosmo.tau_reio()
        loglkl = -0.5 * (tau - self.tau) ** 2 / (self.sigma ** 2)
        return loglkl
