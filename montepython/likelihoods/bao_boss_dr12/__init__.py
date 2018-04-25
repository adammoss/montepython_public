import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class bao_boss_dr12(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # are there conflicting experiments?
        conflicting_experiments = [
            'bao', 'bao_boss', 'bao_known_rs'
            'bao_boss_aniso', 'bao_boss_aniso_gauss_approx']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'conflicting BAO measurments')

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.DM_rdfid_by_rd_in_Mpc = np.array([], 'float64')
        self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.array([], 'float64')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dM(rsfid/rs)':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.DM_rdfid_by_rd_in_Mpc = np.append(
                            self.DM_rdfid_by_rd_in_Mpc, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.append(
                            self.H_rd_by_rdfid_in_km_per_s_per_Mpc, float(this_line[2]))

        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))

        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.cov_data)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # define array for  values of D_M_diff = D_M^th - D_M^obs and H_diff = H^th - H^obs,
        # ordered by redshift bin (z=[0.38, 0.51, 0.61]) as following:
        # data_array = [DM_diff(z=0.38), H_diff(z=0.38), DM_diff(z=0.51), .., .., ..]
        data_array = np.array([], 'float64')

        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # sound horizon at baryon drag rs_d, theoretical prediction
        for i in range(self.num_bins):
            DM_at_z = cosmo.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo.Hubble(self.z[i]) * conts.c / 1000.0
            rd = cosmo.rs_drag() * self.rs_rescale

            theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.rd_fid_in_Mpc
            theo_H_rd_by_rdfid = H_at_z * rd / self.rd_fid_in_Mpc

            # calculate difference between the sampled point and observations
            DM_diff = theo_DM_rdfid_by_rd_in_Mpc - self.DM_rdfid_by_rd_in_Mpc[i]
            H_diff = theo_H_rd_by_rdfid - self.H_rd_by_rdfid_in_km_per_s_per_Mpc[i]

            # save to data array
            data_array = np.append(data_array, DM_diff)
            data_array = np.append(data_array, H_diff)

        # compute chi squared
        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = np.dot(np.dot(data_array,inv_cov_data),data_array)

        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl
