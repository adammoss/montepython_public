from montepython.likelihood_class import Likelihood

class KiDS450_lensing(Likelihood):

    # initialization routine
    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': '1.'})

    # compute likelihood
    def loglkl(self, cosmo, data):

        if self.Nbins == 3:
            lkl = -0.5 * pow(((cosmo.sigma8() * pow(cosmo.Omega_m()/self.Omega_m_ref,self.Omega_m_index) - self.bestfit)_3bin/self.sigma_3bin),2)
        elif self.Nbins == 2:
            lkl = -0.5 * pow(((cosmo.sigma8() * pow(cosmo.Omega_m()/self.Omega_m_ref,self.Omega_m_index) - self.bestfit_2bin)/self.sigma_2bin),2)
        else:
            raise io_mp.LikelihoodError("Number of bins set to an invalid number. Nbins must be 2 or 3.)
            
        return lkl
