import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import scipy.constants as conts


class fake_desi_euclid_bao(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # exclude the isotropic CMASS experiment when the anisotrpic
        # measurement is also used
        exclude_isotropic_CMASS = False

        conflicting_experiments = [
            'bao_boss_aniso', 'bao_boss_aniso_gauss_approx']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                exclude_isotropic_CMASS = True

        if exclude_isotropic_CMASS:
            warnings.warn("excluding isotropic CMASS measurement")
            if not hasattr(self, 'exclude') or self.exclude == None:
                self.exclude = ['CMASS']
            else:
                self.exclude.append('CMASS')

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')
        self.data1 = np.array([], 'float64')
        self.data2 = np.array([], 'float64')
        self.invcov11 = np.array([], 'float64')
        self.invcov22 = np.array([], 'float64')
        self.invcov12 = np.array([], 'float64')

        # read redshifts and mock data points/errors
        self.fid_values_exist = False
        if os.path.exists(os.path.join(self.data_directory, self.fiducial_file)):
            self.fid_values_exist = True
            with open(os.path.join(self.data_directory, self.fiducial_file), 'r') as filein:
                for line in filein:
                    if line.strip() and line.find('#') == -1:
                        # the first entry of the line is the identifier
                        this_line = line.split()
                        # insert into array if this id is not manually excluded
                        if not this_line[0] in self.exclude:
                            self.z = np.append(self.z, float(this_line[1]))
                            type = int(this_line[-1])
                            self.type = np.append(self.type, type)
                            if type != 8:
                                self.data = np.append(self.data, float(this_line[2]))
                                self.error = np.append(self.error, float(this_line[3]))
                                self.data1 = np.append(self.data1, 0.)
                                self.data2 = np.append(self.data2, 0.)
                                self.invcov11 = np.append(self.invcov11, 0.)
                                self.invcov22 = np.append(self.invcov22, 0.)
                                self.invcov12 = np.append(self.invcov12, 0.)
                            else:
                                self.data = np.append(self.data, 0.)
                                self.error = np.append(self.error, 0.)
                                self.data1 = np.append(self.data1, float(this_line[2]))
                                self.data2 = np.append(self.data2, float(this_line[3]))
                                self.invcov11 = np.append(self.invcov11, float(this_line[4]))
                                self.invcov22 = np.append(self.invcov22, float(this_line[5]))
                                self.invcov12 = np.append(self.invcov12, float(this_line[6]))

            # number of data points
            self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # Write fiducial model spectra if needed (return an imaginary number in
        # that case)
        # If not, compute chi2 and proceed

        # If writing fiducial model is needed: read sensitivity (relative errors)
        if self.fid_values_exist is False:

            # open file where fiducial model will be written and write header
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'w')
            fid_file.write('# Fiducial parameters')
            for key, value in data.mcmc_parameters.iteritems():
                fid_file.write(', %s = %.5g' % (
                    key, value['current']*value['scale']))
            fid_file.write('\n')

            # open sensititivy file and ready relative errors
            if os.path.exists(os.path.join(self.data_directory, self.sensitivity)):

                sensitivity = np.loadtxt(os.path.join(os.path.join(self.data_directory, self.sensitivity)))
                self.num_points = np.shape(sensitivity)[0]

                relative_error = np.array([], 'float64')
                relative_invcov11 = np.array([], 'float64')
                relative_invcov22 = np.array([], 'float64')
                relative_invcov12 = np.array([], 'float64')

                for i in range(self.num_points):
                    self.z = np.append(self.z, sensitivity[i,0])
                    self.type = np.append(self.type, self.error_type)
                    if self.type[i] != 8:
                        relative_error = np.append(relative_error, 0.01 * sensitivity[i,self.error_column])
                    else:
                        relative_invcov11 = np.append(relative_invcov11, sensitivity[i,1])
                        relative_invcov22 = np.append(relative_invcov22, sensitivity[i,2])
                        relative_invcov12 = np.append(relative_invcov12, sensitivity[i,3])
            else:
                raise io_mp.LikelihoodError("Could not find file ",self.sensitivity)

        # in all cases: initialise chi2 and compute observables:
        # angular distance da, radial distance dr,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # Hubble parameter in km/s/Mpc
        chi2 = 0.
        for i in range(self.num_points):

            da = cosmo.angular_distance(self.z[i])
            dr = self.z[i] / cosmo.Hubble(self.z[i])
            dv = pow(da * da * (1 + self.z[i]) * (1 + self.z[i]) * dr, 1. / 3.)
            rs = cosmo.rs_drag()
            Hz = cosmo.Hubble(self.z[i]) * conts.c / 1000.0

            if self.type[i] == 3:
                theo = dv / rs

            elif self.type[i] == 4:
                theo = dv

            elif self.type[i] == 5:
                theo = da / rs

            elif self.type[i] == 6:
                theo = 1. / cosmo.Hubble(self.z[i]) / rs

            elif self.type[i] == 7:
                theo = rs / dv

            elif self.type[i] == 8:
                theo1 = Hz*rs
                theo2 = da/rs

            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "BAO data type %s " % self.type[i] +
                    "in %d-th line not understood" % i)

            # if the fiducial model already exists: compute chi2
            if self.fid_values_exist is True:
                if self.type[i] != 8:
                    chi2 += ((theo - self.data[i]) / self.error[i]) ** 2
                else:
                    chi2 += self.invcov11[i]*pow(theo1-self.data1[i],2)+self.invcov22[i]*pow(theo2-self.data2[i],2)+self.invcov12[i]*2.*(theo1-self.data1[i])*(theo2-self.data2[i])

            # if the fiducial model does not exists: write fiducial model
            else:
                if self.type[i] != 8:
                    sigma = theo * relative_error[i]
                    fid_file.write(self.nickname)
                    fid_file.write("   %.8g  %.8g  %.8g %5d \n" % (self.z[i], theo, sigma, self.type[i]))
                else:
                    invcovmat11=relative_invcov11[i]/theo1/theo1
                    print i,relative_invcov11[i],invcovmat11
                    invcovmat22=relative_invcov22[i]/theo2/theo2
                    invcovmat12=relative_invcov12[i]/theo1/theo2
                    fid_file.write(self.nickname)
                    fid_file.write("   %7.8g  %16.8g  %16.8g  %16.8e  %16.8e  %16.8e  %5d\n"
                                   % (self.z[i], theo1, theo2, invcovmat11, invcovmat22, invcovmat12, self.type[i]))


        # Exit after writing fiducial file
        # (return an imaginary number to let the sampler know that fiducial models were just created)
        if self.fid_values_exist is False:
            print '\n'
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        # Otherise, exit normally, returning ln(L)
        lkl = - 0.5 * chi2
        return lkl
