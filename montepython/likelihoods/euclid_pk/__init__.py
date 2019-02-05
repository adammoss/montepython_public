from montepython.likelihood_class import Likelihood
import os
import numpy as np
import warnings
from numpy import newaxis as na
from math import exp, log, pi, log10
import io_mp
import scipy.interpolate

# 'TS;' marks modifications by Tim Sprenger in 2017
# to account for the difference between mu_fid and mu_th and V_fid and V_th
# and to make all volumes comoving
# In addition a new linear cutoff function and the nonlinear velocity dispersion sigma_NL have been added
# The theoretical error has been rescaled

# Now P_cb instead of P_m is used and only linear quantities are used for RSD


class euclid_pk(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

	#TS; new k_cut
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.k_cut(self.zmax)})

        # TS;Compute non-linear power spectrum if requested
        if self.use_halofit:
            	self.need_cosmo_arguments(data, {'non linear':'halofit'})
		print("Using halofit")

        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
        self.fid_values_exist = False
        if os.path.exists(fid_file_path):
            self.fid_values_exist = True

        #################
        # find number of galaxies for each mean redshift value
        #################

        # Deduce the dz step from the number of bins and the edge values of z
        self.dz = (self.zmax-self.zmin)/(self.nbin-1.)

        # Compute the number of galaxies for each \bar z
        # For this, one needs dn/dz TODO
        # then n_g(\bar z) = int_{\bar z - dz/2}^{\bar z + dz/2} dn/dz dz

        # self.z_mean will contain the central values
        self.z_mean = np.linspace(self.zmin, self.zmax, num=self.nbin)

        # Store the z edge values
        self.z_edges = np.linspace(
            self.zmin-self.dz/2., self.zmax+self.dz/2,
            num=self.nbin+1)

        # Store the total vector z, with edges + mean
        self.z = np.linspace(
            self.zmin-self.dz/2., self.zmax+self.dz/2.,
            num=2*self.nbin+1)

        # Define the k values for the integration (from kmin to kmax), at which
        # the spectrum will be computed (and stored for the fiducial model)
        # k_size is deeply arbitrary here, TODO
        self.k_fid = np.logspace(
            log10(self.kmin), log10(self.k_cut(self.zmax)), num=self.k_size)

        # TS; Define the mu scale
        self.mu_fid = np.linspace(-1, 1, self.mu_size)

        ################
        # Noise spectrum
        ################
	# TS; number counts from model 1 in Pozzetti et al. (1603.01453)

        self.n_g = np.zeros(self.nbin, 'float64')

        #self.n_g = np.array([6844.945, 7129.45,
        #                     7249.912, 7261.722,
        #                     7203.825, 7103.047,
        #                     6977.571, 6839.546,
        #                     6696.957, 5496.988,
        #                     4459.240, 3577.143,
        #                     2838.767, 2229.282,
        #                     1732.706, 1333.091])
        #self.n_g = self.n_g * self.fsky * 41253. * self.efficiency
        self.n_g = np.array([4825,4112,3449,2861,2357,1933,1515,1140,861,654,499,382,295])
        self.n_g = self.n_g * self.fsky * 41253. * self.dz

	# TS; Ntot output
	#print("\nEuclid: Number of detected galaxies in each redshift bin:")
	#for index_z in xrange(self.nbin):
	#	print("z-bin[" + str(self.z_mean[index_z]-self.dz/2.) + "," + str(self.z_mean[index_z]+self.dz/2.) + "]: \tN = %.4g" % (self.n_g[index_z]) + " ,\t b = %.4g" % (self.b[index_z]))
	#N_tot = np.sum(self.n_g)
	#print("Total number of detected galaxies: N = %.4g\n" % (N_tot))

        # If the file exists, initialize the fiducial values, the spectrum will
        # be read first, with k_size values of k and nbin values of z. Then,
        # H_fid and D_A fid will be read (each with nbin values).
        self.fid_values_exist = False
        self.pk_nl_fid = np.zeros((self.k_size, 2*self.nbin+1), 'float64')
        if self.use_linear_rsd:
            self.pk_lin_fid = np.zeros((self.k_size, 2*self.nbin+1), 'float64')
        self.H_fid = np.zeros(2*self.nbin+1, 'float64')
        self.D_A_fid = np.zeros(2*self.nbin+1, 'float64')
        self.sigma_r_fid = np.zeros(self.nbin, 'float64')
        self.V_fid = np.zeros(self.nbin, 'float64')
        self.b_fid = np.zeros(self.nbin, 'float64')

        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
        if os.path.exists(fid_file_path):
            self.fid_values_exist = True
            with open(fid_file_path, 'r') as fid_file:
                line = fid_file.readline()
                while line.find('#') != -1:
                    line = fid_file.readline()
                while (line.find('\n') != -1 and len(line) == 1):
                    line = fid_file.readline()
                for index_k in xrange(self.k_size):
                    for index_z in xrange(2*self.nbin+1):
                        if self.use_linear_rsd:
                            self.pk_nl_fid[index_k, index_z] = float(line.split()[0])
                            self.pk_lin_fid[index_k, index_z] = float(line.split()[1])
                        else:
                            self.pk_nl_fid[index_k, index_z] = float(line)
                        line = fid_file.readline()
                for index_z in xrange(2*self.nbin+1):
                    self.H_fid[index_z] = float(line.split()[0])
                    self.D_A_fid[index_z] = float(line.split()[1])
                    line = fid_file.readline()
                for index_z in xrange(self.nbin):
                    self.sigma_r_fid[index_z] = float(line.split()[0])
		    #TS; load fiducial volume
		    self.V_fid[index_z] = float(line.split()[1])
		    #TS; load fiducial bias
		    self.b_fid[index_z] = float(line.split()[2])
                    line = fid_file.readline()
		#TS; load fiducial sigma_NL
		self.sigma_NL_fid = float(line)
        # Else the file will be created in the loglkl() function.
        return

    # Galaxy distribution, returns the function D(z) from the notes
    def galaxy_distribution(self, z):

        zmean = 0.9
        z0 = zmean/1.412

        galaxy_dist = z**2*exp(-(z/z0)**(1.5))

        return galaxy_dist

    # TS; made kmax a function potentially dependent on redshift
    def k_cut(self, z,h=0.6693,n_s=0.9619):
	kcut = self.kmax*h
	# compute kmax according to highest redshift linear cutoff (1509.07562v2)
	if self.use_zscaling:
		kcut *= pow(1.+z,2./(2.+n_s))
	return kcut

    def loglkl(self, cosmo, data):
        # First thing, recover the angular distance and Hubble factor for each
        # redshift
        H = np.zeros(2*self.nbin+1, 'float64')
        D_A = np.zeros(2*self.nbin+1, 'float64')
        r = np.zeros(2*self.nbin+1, 'float64')

        # H is incidentally also dz/dr
        r, H = cosmo.z_of_r(self.z)
        for i in xrange(len(D_A)):
            D_A[i] = cosmo.angular_distance(self.z[i])

        # Compute sigma_r = dr(z)/dz sigma_z with sigma_z = 0.001(1+z)
        sigma_r = np.zeros(self.nbin, 'float64')
        for index_z in xrange(self.nbin):
            sigma_r[index_z] = 0.001*(1.+self.z_mean[index_z])/H[2*index_z+1]

	# TS; Option: nuisance sigma_NL in Mpc = nonlinear dispersion scale of RSD (1405.1452v2)
	sigma_NL = 0.0	# fiducial would be 7 but when kept constant that is more constraining than keeping 0
	if 'sigma_NL' in self.use_nuisance:
		sigma_NL = data.mcmc_parameters['sigma_NL']['current']*data.mcmc_parameters['sigma_NL']['scale']

        # At the center of each bin, compute the bias function, simply taken
        # as sqrt(z_mean+1)
	if 'beta_0^Euclid' in self.use_nuisance:
            b = pow(1.+self.z_mean,0.5*data.mcmc_parameters['beta_1^Euclid']['current']*data.mcmc_parameters['beta_1^Euclid']['scale'])*data.mcmc_parameters['beta_0^Euclid']['current']*data.mcmc_parameters['beta_0^Euclid']['scale']
	else:
            b = np.sqrt(1.+self.z_mean)

        # Compute V_survey, for each given redshift bin,
        # which is the volume of a shell times the sky coverage:
	# TS; no more need for self., now comoving, exact integral solution
        V_survey = np.zeros(self.nbin, 'float64')
        for index_z in xrange(self.nbin):
            V_survey[index_z] = 4./3.*pi*self.fsky*(
                r[2*index_z+2]**3-r[2*index_z]**3)

        # If the fiducial model does not exist, recover the power spectrum and
        # store it, then exit.
        if self.fid_values_exist is False:
            pk = np.zeros((self.k_size, 2*self.nbin+1), 'float64')
            if self.use_linear_rsd:
                pk_lin = np.zeros((self.k_size, 2*self.nbin+1), 'float64')
            fid_file_path = os.path.join(
                self.data_directory, self.fiducial_file)
            with open(fid_file_path, 'w') as fid_file:
                fid_file.write('# Fiducial parameters')
                for key, value in data.mcmc_parameters.iteritems():
                    fid_file.write(', %s = %.5g' % (
                        key, value['current']*value['scale']))
                fid_file.write('\n')
                for index_k in xrange(self.k_size):
                    for index_z in xrange(2*self.nbin+1):
                        pk[index_k, index_z] = cosmo.pk_cb(
                            self.k_fid[index_k], self.z[index_z])
                        if self.use_linear_rsd:
                            pk_lin[index_k, index_z] = cosmo.pk_cb_lin(
                                self.k_fid[index_k], self.z[index_z])
                            fid_file.write('%.8g %.8g\n' % (pk[index_k, index_z], pk_lin[index_k, index_z]))
                        else:
                            fid_file.write('%.8g\n' % pk[index_k, index_z])
                for index_z in xrange(2*self.nbin+1):
                    fid_file.write('%.8g %.8g\n' % (H[index_z], D_A[index_z]))
                for index_z in xrange(self.nbin):
		# TS; save fiducial survey volume V_fid
                    fid_file.write('%.8g %.8g %.8g\n' % (sigma_r[index_z], V_survey[index_z], b[index_z]))
		# TS; save fiducial sigma_NL
		fid_file.write('%.8g\n' % sigma_NL)
            print '\n'
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        # NOTE: Many following loops will be hidden in a very specific numpy
        # expression, for (a more than significant) speed-up. All the following
        # loops keep the same pattern.  The colon denotes the whole range of
        # indices, so beta_fid[:,index_z] denotes the array of length
        # self.k_size at redshift z[index_z]

        # Compute the beta_fid function, for observed spectrum,
        # beta_fid(k_fid,z) = 1/2b(z) * d log(P_nl_fid(k_fid,z))/d log a
        #                   = -1/2b(z)* (1+z) d log(P_nl_fid(k_fid,z))/dz

        if self.use_linear_rsd:
            beta_fid = -0.5/self.b_fid*(1+self.z_mean)*np.log(
                self.pk_lin_fid[:, 2::2]/self.pk_lin_fid[:, :-2:2])/self.dz
        else:
            beta_fid = -0.5/self.b_fid*(1+self.z_mean)*np.log(
                self.pk_nl_fid[:, 2::2]/self.pk_nl_fid[:, :-2:2])/self.dz

        # Compute the tilde P_fid(k_ref,z,mu) = H_fid(z)/D_A_fid(z)**2 ( 1 + beta_fid(k_fid,z)mu^2)^2 P_nl_fid(k_fid,z)exp ( -k_fid^2 mu^2 sigma_r_fid^2)
        self.tilde_P_fid = np.zeros((self.k_size, self.nbin, self.mu_size),
                                    'float64')

        self.tilde_P_fid = self.H_fid[na, 1::2, na]/(
            self.D_A_fid[na, 1::2, na])**2*self.b_fid[na,:,na]**2*(
                1. + beta_fid[:, :, na] * self.mu_fid[na, na, :]**2)**2 * (
            self.pk_nl_fid[:, 1::2, na]) * np.exp(
                -self.k_fid[:, na, na]**2 * self.mu_fid[na, na, :]**2 *
                (self.sigma_r_fid[na, :, na]**2+self.sigma_NL_fid**2))


        ######################
        # TH PART
        ######################
        # Compute values of k based on k_fid (ref in paper), with formula (33 has to be corrected):
        # k^2 = ( (1-mu_fid^2) D_A_fid(z)^2/D_A(z)^2 + mu_fid^2 H(z)^2/H_fid(z)^2) k_fid ^ 2
        # So k = k (k_ref,z,mu)
	# TS; changed mu -> mu_fid
        self.k = np.zeros((self.k_size,2*self.nbin+1,self.mu_size),'float64')
        for index_k in xrange(self.k_size):
            for index_z in xrange(2*self.nbin+1):
                self.k[index_k,index_z,:] = np.sqrt((1.-self.mu_fid[:]**2)*self.D_A_fid[index_z]**2/D_A[index_z]**2 + self.mu_fid[:]**2*H[index_z]**2/self.H_fid[index_z]**2 )*self.k_fid[index_k]

	# TS; Compute values of mu based on mu_fid with
	# mu^2 = mu_fid^2 / (mu_fid^2 + ((H_fid*D_A_fid)/(H*D_A))^2)*(1 - mu_fid^2))
	self.mu = np.zeros((self.nbin,self.mu_size),'float64')
	for index_z in xrange(self.nbin):
		self.mu[index_z,:] = np.sqrt(self.mu_fid[:]**2/(self.mu_fid[:]**2 + ((self.H_fid[2*index_z+1]*self.D_A_fid[2*index_z+1])/(D_A[2*index_z+1]*H[2*index_z+1]))**2 * (1.-self.mu_fid[:]**2)))

        # Recover the non-linear power spectrum from the cosmological module on all
        # the z_boundaries, to compute afterwards beta. This is pk_nl_th from the
        # notes.
        pk_nl_th = np.zeros((self.k_size,2*self.nbin+1,self.mu_size),'float64')
        if self.use_linear_rsd:
            pk_lin_th = np.zeros((self.k_size,2*self.nbin+1,self.mu_size),'float64')

        # The next line is the bottleneck.
        # TODO: the likelihood could be sped up if this could be vectorised, either here,
        # or inside classy where there are three loops in the function get_pk
        # (maybe with a different strategy for the arguments of the function)
        pk_nl_th = cosmo.get_pk_cb(self.k,self.z,self.k_size,2*self.nbin+1,self.mu_size)
        if self.use_linear_rsd:
            pk_lin_th = cosmo.get_pk_cb_lin(self.k,self.z,self.k_size,2*self.nbin+1,self.mu_size)

        # Define the alpha function, that will characterize the theoretical
        # uncertainty. (TS; = theoretical error envelope)
	# TS; introduced new envelope (0:optimistic 1:pessimistic for ~2023), (for old envelope see commented)
        self.alpha = np.zeros((self.k_size,self.nbin,self.mu_size),'float64')
	th_c1 = 0.75056
	th_c2 = 1.5120
	th_a1 = 0.014806
	th_a2 = 0.022047
       	for index_z in xrange(self.nbin):
	    k_z = cosmo.h()*pow(1.+self.z_mean[index_z],2./(2.+cosmo.n_s()))
	    for index_mu in xrange(self.mu_size):
	        for index_k in xrange(self.k_size):
	            if self.k[index_k,2*index_z+1,index_mu]/k_z<0.3:
	                self.alpha[index_k,index_z,index_mu] = th_a1*np.exp(th_c1*np.log10(self.k[index_k,2*index_z+1,index_mu]/k_z))
	            else:
	                self.alpha[index_k,index_z,index_mu] = th_a2*np.exp(th_c2*np.log10(self.k[index_k,2*index_z+1,index_mu]/k_z))

	# TS; Define fractional theoretical error variance R/P^2
	self.R_var = np.zeros((self.k_size,self.nbin,self.mu_size),'float64')
	for index_k in xrange(self.k_size):
	    for index_z in xrange(self.nbin):
	        self.R_var[index_k,index_z,:] = self.V_fid[index_z]/(2.*np.pi)**2*self.k_CorrLength_hMpc*cosmo.h()/self.z_CorrLength*self.dz*self.k_fid[index_k]**2*self.alpha[index_k,index_z,:]**2


	# TS; neutrino error obsolete since halofit update; corresponding lines were deleted

        # Compute the beta function for nl,
        # beta(k,z) = 1/2b(z) * d log(P_nl_th (k,z))/d log a
        #           = -1/2b(z) *(1+z) d log(P_nl_th (k,z))/dz
        beta_th = np.zeros((self.k_size,self.nbin,self.mu_size),'float64')
        for index_k in xrange(self.k_size):
            for index_z in xrange(self.nbin):
                if self.use_linear_rsd:
                    beta_th[index_k,index_z,:] = -1./(2.*b[index_z]) * (1.+self.z_mean[index_z]) * np.log(pk_lin_th[index_k,2*index_z+2,:]/pk_lin_th[index_k,2*index_z,:])/(self.dz)
                else:
                    beta_th[index_k,index_z,:] = -1./(2.*b[index_z]) * (1.+self.z_mean[index_z]) * np.log(pk_nl_th[index_k,2*index_z+2,:]/pk_nl_th[index_k,2*index_z,:])/(self.dz)

        # Compute \tilde P_th(k,mu,z) = H(z)/D_A(z)^2 * (1 + beta(z,k) mu^2)^2 P_nl_th(k,z) exp(-k^2 mu^2 (sigma_r^2+sigma_NL^2))
	# TS; mu -> self.mu, added sigma_NL contribution
        self.tilde_P_th = np.zeros( (self.k_size,self.nbin,self.mu_size), 'float64')
        for index_k in xrange(self.k_size):
            for index_z in xrange(self.nbin):
                self.tilde_P_th[index_k,index_z,:] = H[2*index_z+1]/(D_A[2*index_z+1]**2) * b[index_z]**2*(1. + beta_th[index_k,index_z,:]*self.mu[index_z,:]*self.mu[index_z,:])**2* pk_nl_th[index_k,2*index_z+1,:]*np.exp(-self.k[index_k,2*index_z+1,:]**2*self.mu[index_z,:]**2*(sigma_r[index_z]**2+sigma_NL**2))

        # Shot noise spectrum:
	# TS; Removed necessity of specifying a nuisance P_shot (not used in standard)
	# and inserted new self.V_fid
        self.P_shot = np.zeros( (self.nbin),'float64')
        for index_z in xrange(self.nbin):
        	if 'P_shot' in self.use_nuisance:
			self.P_shot[index_z] = self.H_fid[2*index_z+1]/(self.D_A_fid[2*index_z+1]**2)*(data.mcmc_parameters['P_shot']['current']*data.mcmc_parameters['P_shot']['scale'] + self.V_fid[index_z]/self.n_g[index_z])
        	else:
			self.P_shot[index_z] = self.H_fid[2*index_z+1]/(self.D_A_fid[2*index_z+1]**2)*(self.V_fid[index_z]/self.n_g[index_z])

        # finally compute chi2, for each z_mean
	if self.use_zscaling:
		# TS; reformulated loops to include z-dependent kmax, mu -> mu_fid
        	chi2 = 0.0
		index_kmax = 0
		delta_mu = self.mu_fid[1] - self.mu_fid[0] # equally spaced
		integrand_low = 0.0
		integrand_hi = 0.0

		for index_z in xrange(self.nbin):
			# uncomment printers to show chi2 contribution from single bins
			#printer1 = chi2*delta_mu
			# TS; uncomment to display max. kmin (used to infer kmin~0.02):
			#kmin: #print("z=" + str(self.z_mean[index_z]) + " kmin=" + str(34.56/r[2*index_z+1]) + "\tor " + str(6.283/(r[2*index_z+2]-r[2*index_z])))
			for index_k in xrange(1,self.k_size):
				if ((self.k_cut(self.z_mean[index_z],cosmo.h(),cosmo.n_s())-self.k_fid[self.k_size-index_k]) > -1.e-6):
					index_kmax = self.k_size-index_k
					break
			integrand_low = self.integrand(0,index_z,0)*.5
			for index_k in xrange(1,index_kmax+1):
				integrand_hi = self.integrand(index_k,index_z,0)*.5
				chi2 += (integrand_hi+integrand_low)*.5*(self.k_fid[index_k]-self.k_fid[index_k-1])
				integrand_low = integrand_hi
			chi2 += integrand_low*(self.k_cut(self.z_mean[index_z],cosmo.h(),cosmo.n_s())-self.k_fid[index_kmax])
			for index_mu in xrange(1,self.mu_size-1):
				integrand_low = self.integrand(0,index_z,index_mu)
				for index_k in xrange(1,index_kmax+1):
					integrand_hi = self.integrand(index_k,index_z,index_mu)
					chi2 += (integrand_hi+integrand_low)*.5*(self.k_fid[index_k]-self.k_fid[index_k-1])
					integrand_low = integrand_hi
				chi2 += integrand_low*(self.k_cut(self.z_mean[index_z],cosmo.h(),cosmo.n_s())-self.k_fid[index_kmax])
			integrand_low = self.integrand(0,index_z,self.mu_size-1)*.5
			for index_k in xrange(1,index_kmax+1):
				integrand_hi = self.integrand(index_k,index_z,self.mu_size-1)*.5
				chi2 += (integrand_hi+integrand_low)*.5*(self.k_fid[index_k]-self.k_fid[index_k-1])
				integrand_low = integrand_hi
			chi2 += integrand_low*(self.k_cut(self.z_mean[index_z],cosmo.h(),cosmo.n_s())-self.k_fid[index_kmax])
			#printer2 = chi2*delta_mu-printer1
			#print("%s\t%s" % (self.z_mean[index_z], printer2))
		chi2 *= delta_mu

	else:
            # TS; original code with integrand() -> array_integrand()
            chi2 = 0.0
            mu_integrand_lo,mu_integrand_hi = 0.0,0.0
            k_integrand  = np.zeros(self.k_size,'float64')
            for index_z in xrange(self.nbin):
                k_integrand = self.array_integrand(index_z,0)
                mu_integrand_hi = np.sum((k_integrand[1:] + k_integrand[0:-1])*.5*(self.k_fid[1:] - self.k_fid[:-1]))
                for index_mu in xrange(1,self.mu_size):
                    mu_integrand_lo = mu_integrand_hi
                    mu_integrand_hi = 0
                    k_integrand = self.array_integrand(index_z,index_mu)
                    mu_integrand_hi = np.sum((k_integrand[1:] + k_integrand[0:-1])*.5*(self.k_fid[1:] - self.k_fid[:-1]))
                    # TS; mu -> mu_fid
                    chi2 += (mu_integrand_hi + mu_integrand_lo)/2.*(self.mu_fid[index_mu] - self.mu_fid[index_mu-1])

        if 'beta_0^Euclid' in self.use_nuisance:
            chi2 += ((data.mcmc_parameters['beta_0^Euclid']['current']*data.mcmc_parameters['beta_0^Euclid']['scale']-1.)/self.bias_accuracy)**2
            chi2 += ((data.mcmc_parameters['beta_1^Euclid']['current']*data.mcmc_parameters['beta_1^Euclid']['scale']-1.)/self.bias_accuracy)**2

        return - chi2/2.

    # TS; V_fid, index_k: index->argument, rescaled theoretical error
    # and added triggers to avoid computation of zero theoretical error
    def integrand(self,index_k,index_z,index_mu):
        if not self.UseTheoError:
            return (self.V_fid[index_z]/2.)*self.k_fid[index_k]**2/(2.*pi)**2*((self.tilde_P_th[index_k,index_z,index_mu] - self.tilde_P_fid[index_k,index_z,index_mu])**2/((self.tilde_P_th[index_k,index_z,index_mu] + self.P_shot[index_z])**2))
        return (self.V_fid[index_z]/2.)*self.k_fid[index_k]**2/(2.*pi)**2*((self.tilde_P_th[index_k,index_z,index_mu] - self.tilde_P_fid[index_k,index_z,index_mu])**2/((self.tilde_P_th[index_k,index_z,index_mu] + self.P_shot[index_z])**2 + self.R_var[index_k,index_z,index_mu]*self.tilde_P_th[index_k,index_z,index_mu]**2))
    def array_integrand(self,index_z,index_mu):
        if not self.UseTheoError:
            return (self.V_fid[index_z]/2.)*self.k_fid[:]**2/(2.*pi)**2*((self.tilde_P_th[:,index_z,index_mu] - self.tilde_P_fid[:,index_z,index_mu])**2/((self.tilde_P_th[:,index_z,index_mu] + self.P_shot[index_z])**2))
        return (self.V_fid[index_z]/2.)*self.k_fid[:]**2/(2.*pi)**2*((self.tilde_P_th[:,index_z,index_mu] - self.tilde_P_fid[:,index_z,index_mu])**2/((self.tilde_P_th[:,index_z,index_mu] + self.P_shot[index_z])**2 + self.R_var[:,index_z,index_mu]*self.tilde_P_th[:,index_z,index_mu]**2))
