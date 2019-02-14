from montepython.likelihood_class import Likelihood
import os
import numpy as np
import warnings
from numpy import newaxis as na
from math import exp, log, pi, log10
import scipy.integrate
# Created by Tim Sprenger in 2017 based on euclid_pk

class ska1_pk(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.k_cut(self.zmax)})

        # Compute non-linear power spectrum if requested
	if self.use_halofit:
            	self.need_cosmo_arguments(data, {'non linear':'halofit'})
		print("Using halofit")

        # Deduce the dz step from the number of bins and the edge values of z
        self.dz = (self.zmax-self.zmin)/self.nbin

	# Compute new zmin and zmax which are bin centers
	# Need to be defined as edges if zmin can be close to z=0
	self.zmin += self.dz/2.
	self.zmax -= self.dz/2.

        # self.z_mean will contain the central values
        self.z_mean = np.linspace(self.zmin, self.zmax, num=self.nbin)

        # Store the total vector z, with edges + mean
        self.z = np.linspace(
            self.zmin-self.dz/2., self.zmax+self.dz/2.,
            num=2*self.nbin+1)

        # Compute the number of galaxies for each \bar z
        # N_g(\bar z) = int_{\bar z - dz/2}^{\bar z + dz/2} dn/dz dz
	# dn/dz fit formula from 1412.4700v2: 10^c1*z^c2*e^-c3z
	# dn/dz = number of galaxies per redshift and deg^2
	# old approx.: self.N_g = pow(10.,self.c1)*pow(self.z_mean,self.c2)*np.exp(-self.c3*self.z_mean) * self.dz * self.skycov
	self.N_g = np.zeros(self.nbin)
	N_tot = 0.0
	for index_z in xrange(self.nbin):
		self.N_g[index_z], error = scipy.integrate.quad(self.dndz, self.z_mean[index_z]-self.dz/2., self.z_mean[index_z]+self.dz/2.)
		assert error/self.N_g[index_z] <= 0.001, ("dndz integration error is bigger than 0.1%")
		N_tot += self.N_g[index_z]

	# Ntot output
	#print("\nSKA1: Number of detected galaxies and bias in each redshift bin:")
	#for index_z in xrange(self.nbin):
	#	print("z-bin[" + str(self.z_mean[index_z]-self.dz/2.) + "," + str(self.z_mean[index_z]+self.dz/2.) + "]: \tN = %.4g" % (self.N_g[index_z]) + " ,\t b = %.4g" % (b[index_z]))
	#print("Total number of detected galaxies: N = %.4g\n" % (N_tot))

        # Define the k values for the integration (from kmin to kmax), at which
        # the spectrum will be computed (and stored for the fiducial model)
        self.k_fid = np.logspace(
            log10(self.kmin), log10(self.k_cut(self.zmax)), num=self.k_size)

        # Define the mu scale
        self.mu_fid = np.linspace(-1, 1, self.mu_size)

        # If the file exists, initialize the fiducial values, the spectrum will
        # be read first, with k_size values of k and nbin values of z. Then,
        # H_fid and D_A fid will be read (each with nbin values).
	# Then V_fid, b_fid and the fiducial errors on real space coordinates follow.
        self.fid_values_exist = False
        self.pk_nl_fid = np.zeros((self.k_size, 2*self.nbin+1), 'float64')
        if self.use_linear_rsd:
            self.pk_lin_fid = np.zeros((self.k_size, 2*self.nbin+1), 'float64')
        self.H_fid = np.zeros(2*self.nbin+1, 'float64')
        self.D_A_fid = np.zeros(2*self.nbin+1, 'float64')
        self.V_fid = np.zeros(self.nbin, 'float64')
        self.b_fid = np.zeros(self.nbin, 'float64')
	self.sigma_A_fid = np.zeros(self.nbin, 'float64')
	self.sigma_B_fid = np.zeros(self.nbin, 'float64')

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
                    self.V_fid[index_z] = float(line.split()[0])
                    self.b_fid[index_z] = float(line.split()[1])
                    line = fid_file.readline()
                for index_z in xrange(self.nbin):
                    self.sigma_A_fid[index_z] = float(line.split()[0])
                    self.sigma_B_fid[index_z] = float(line.split()[1])
                    line = fid_file.readline()
		self.sigma_NL_fid = float(line)

        # Else the file will be created in the loglkl() function.
        return

    def dndz(self, z):
	# dn/dz/deg^2 fit formula from 1412.4700v2: 10^c1*z^c2*e^-c3z * sky coverage in deg^2
	dndz = pow(10.,self.c1)*pow(z,self.c2)*np.exp(-self.c3*z)*self.skycov
	return dndz

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

        # Compute V_survey, for each given redshift bin, which is the volume of
        # a shell times the sky coverage (only fiducial needed):
	if self.fid_values_exist is False:
        	V_survey = np.zeros(self.nbin, 'float64')
        	for index_z in xrange(self.nbin):
           	 	V_survey[index_z] = 4./3.*pi*self.skycov/41253.*(
               			r[2*index_z+2]**3-r[2*index_z]**3)

        # At the center of each bin, compute the bias function,
        # using fit fromula from 1412.4700v2: c4*e^c5z
	if 'beta_0^SKA1' in self.use_nuisance:
        	b = self.c4*np.exp(self.c5*self.z_mean*data.mcmc_parameters['beta_1^SKA1']['current']*data.mcmc_parameters['beta_1^SKA1']['scale'])*data.mcmc_parameters['beta_0^SKA1']['current']*data.mcmc_parameters['beta_0^SKA1']['scale']
	else:
        	b = self.c4*np.exp(self.c5*self.z_mean)

    	# Compute freq.res. sigma_r = (1+z)^2/H*sigma_nu/nu_21cm, nu in Mhz
	# Compute ang.res. sigma_perp = 1.22(1+z)^2*D_A*lambda_21cm/Baseline, Baseline in km
	# combine into exp(-k^2*(mu^2*(sig_r^2-sig_perp^2)+sig_perp^2)) independent of cosmo
	# used as exp(-k^2*(mu^2*sigma_A+sigma_B)) all fiducial
	if self.fid_values_exist is False:
		sigma_A = np.zeros(self.nbin,'float64')
		sigma_B = np.zeros(self.nbin,'float64')
		sigma_A = ((1.+self.z_mean[:])**2/H[1::2]*self.delta_nu/np.sqrt(8.*np.log(2.))/self.nu0)**2 -(
			1./np.sqrt(8.*np.log(2.))*(1+self.z_mean[:])**2 * D_A[1::2]*2.111e-4/self.Baseline)**2
		sigma_B = (1./np.sqrt(8.*np.log(2.))*(1+self.z_mean[:])**2 * D_A[1::2]*2.111e-4/self.Baseline)**2

	# sigma_NL in Mpc = nonlinear dispersion scale of RSD (1405.1452v2)
	sigma_NL = 0.0	# fiducial would be 7 but when kept constant that is more constraining than keeping 0
	if 'sigma_NL' in self.use_nuisance:
		sigma_NL = data.mcmc_parameters['sigma_NL']['current']*data.mcmc_parameters['sigma_NL']['scale']

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
                    fid_file.write('%.8g %.8g\n' % (V_survey[index_z],b[index_z]))
                for index_z in xrange(self.nbin):
			fid_file.write('%.8g %.8g\n' % (sigma_A[index_z], sigma_B[index_z]))
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

        # Compute the tilde P_fid(k_ref,z,mu) = H_fid(z)/D_A_fid(z)**2 ( 1 + beta_fid(k_fid,z)mu^2)^2 P_nl_fid(k_fid,z)exp(-k_fid^2*(mu_fid^2*sigma_A(z)+sigma_B(z)))
        self.tilde_P_fid = np.zeros((self.k_size, self.nbin, self.mu_size),'float64')
        self.tilde_P_fid = self.H_fid[na, 1::2, na]/(
            	self.D_A_fid[na, 1::2, na])**2*self.b_fid[na,:,na]**2*(
                1. + beta_fid[:, :, na] * self.mu_fid[na, na, :]**2)**2 * (
            	self.pk_nl_fid[:, 1::2, na]) * np.exp(-self.k_fid[:,na,na]**2 *
		(self.mu_fid[na, na, :]**2*(self.sigma_A_fid[na,:,na]+self.sigma_NL_fid**2) + self.sigma_B_fid[na,:,na]))

        ######################
        # TH PART
        ######################
        # Compute values of k based on fiducial values:
        # k^2 = ( (1-mu^2) D_A_fid(z)^2/D_A(z)^2 + mu^2 H(z)^2/H_fid(z)^2) k_fid ^ 2
        self.k = np.zeros((self.k_size,2*self.nbin+1,self.mu_size),'float64')
        for index_k in xrange(self.k_size):
            for index_z in xrange(2*self.nbin+1):
                self.k[index_k,index_z,:] = np.sqrt((1.-self.mu_fid[:]**2)*self.D_A_fid[index_z]**2/D_A[index_z]**2 + self.mu_fid[:]**2*H[index_z]**2/self.H_fid[index_z]**2 )*self.k_fid[index_k]

	# Compute values of mu based on fiducial values:
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

	if self.UseTheoError :
        	# Recover the non_linear scale computed by halofit.
        	#self.k_sigma = np.zeros(2*self.nbin+1, 'float64')
            	#self.k_sigma = cosmo.nonlinear_scale(self.z,2*self.nbin+1)

        	# Define the theoretical error envelope
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

		# Define fractional theoretical error variance R/P^2
		self.R_var = np.zeros((self.k_size,self.nbin,self.mu_size),'float64')
		for index_k in xrange(self.k_size):
	    	    for index_z in xrange(self.nbin):
	                self.R_var[index_k,index_z,:] = self.V_fid[index_z]/(2.*np.pi)**2*self.k_CorrLength_hMpc*cosmo.h()/self.z_CorrLength*self.dz*self.k_fid[index_k]**2*self.alpha[index_k,index_z,:]**2

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

        # Compute \tilde P_th(k,mu,z) = H(z)/D_A(z)^2 * (1 + beta(z,k) mu^2)^2 exp(-k^2 mu^2 sigma_NL^2) P_nl_th(k,z) exp(-k^2 (mu^2 sigma_A + sigma_B))
        self.tilde_P_th = np.zeros( (self.k_size,self.nbin,self.mu_size), 'float64')
        for index_k in xrange(self.k_size):
            for index_z in xrange(self.nbin):
                self.tilde_P_th[index_k,index_z,:] = H[2*index_z+1]/(D_A[2*index_z+1]**2) * b[index_z]**2*(1. + beta_th[index_k,index_z,:]*self.mu[index_z,:]*self.mu[index_z,:])**2*np.exp(-self.k[index_k,2*index_z+1,:]**2*self.mu[index_z,:]**2*sigma_NL**2)* pk_nl_th[index_k,2*index_z+1,:]*np.exp(-self.k_fid[index_k]**2*(self.mu_fid[:]**2*self.sigma_A_fid[index_z] + self.sigma_B_fid[index_z]))

        # Shot noise spectrum
        self.P_shot = np.zeros( (self.nbin),'float64')
        for index_z in xrange(self.nbin):
            self.P_shot[index_z] = self.H_fid[2*index_z+1]/(self.D_A_fid[2*index_z+1]**2)*self.V_fid[index_z]/self.N_g[index_z]

        # finally compute chi2, for each z_mean
	if self.use_zscaling==0:
		# redshift dependent cutoff makes integration more complicated
        	chi2 = 0.0
		index_kmax = 0
		delta_mu = self.mu_fid[1] - self.mu_fid[0] # equally spaced
		integrand_low = 0.0
		integrand_hi = 0.0

		for index_z in xrange(self.nbin):
			# uncomment printers to get contributions from individual redshift bins
			#printer1 = chi2*delta_mu
			# uncomment to display max. kmin (used to infer kmin~0.02):
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
                    chi2 += (mu_integrand_hi + mu_integrand_lo)/2.*(self.mu_fid[index_mu] - self.mu_fid[index_mu-1])

        if 'beta_0^SKA1' in self.use_nuisance:
            chi2 += ((data.mcmc_parameters['beta_0^SKA1']['current']*data.mcmc_parameters['beta_0^SKA1']['scale']-1.)/self.bias_accuracy)**2
            chi2 += ((data.mcmc_parameters['beta_1^SKA1']['current']*data.mcmc_parameters['beta_1^SKA1']['scale']-1.)/self.bias_accuracy)**2

        return - chi2/2.

    def integrand(self,index_k,index_z,index_mu):
        if self.UseTheoError :
            return (self.V_fid[index_z]/2.)*self.k_fid[index_k]**2/(2.*pi)**2*((self.tilde_P_th[index_k,index_z,index_mu] - self.tilde_P_fid[index_k,index_z,index_mu])**2/((self.tilde_P_th[index_k,index_z,index_mu] + self.P_shot[index_z])**2 + self.R_var[index_k,index_z,index_mu]*self.tilde_P_th[index_k,index_z,index_mu]**2))
        return (self.V_fid[index_z]/2.)*self.k_fid[index_k]**2/(2.*pi)**2*((self.tilde_P_th[index_k,index_z,index_mu] - self.tilde_P_fid[index_k,index_z,index_mu])**2/((self.tilde_P_th[index_k,index_z,index_mu] + self.P_shot[index_z])**2))
    def array_integrand(self,index_z,index_mu):
        if self.UseTheoError :
            return (self.V_fid[index_z]/2.)*self.k_fid[:]**2/(2.*pi)**2*((self.tilde_P_th[:,index_z,index_mu] - self.tilde_P_fid[:,index_z,index_mu])**2/((self.tilde_P_th[:,index_z,index_mu] + self.P_shot[index_z])**2 + self.R_var[:,index_z,index_mu]*self.tilde_P_th[:,index_z,index_mu]**2))
        return (self.V_fid[index_z]/2.)*self.k_fid[:]**2/(2.*pi)**2*((self.tilde_P_th[:,index_z,index_mu] - self.tilde_P_fid[:,index_z,index_mu])**2/((self.tilde_P_th[:,index_z,index_mu] + self.P_shot[index_z])**2))

#last line
