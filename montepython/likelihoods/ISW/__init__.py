from montepython.likelihood_class import Likelihood_isw, Likelihood
import os

class ISW(Likelihood_isw):
    """
    This likelihood is used to derive constraints on the ISW effect through
    crosscorrelation of the CMB with 5 galaxy surveys: 2MPZ, Wise x SuperCOSMOS,
    SDSS DR12 photometric, SDSS DR6 QSO, and NVSS. Each survey is divided into
    several redshift bins.

    The C_l of auto- and crosscorrelation are binned in l and fitted to the
    cosmological model. The minimum and maximum l and the number of bins is
    specified in the datafile of each survey and redshift bin.

    Fit parameters: ISW amplitude A_ISW, 15 bias parameters (one for each redshift bin of each catalog)
    """
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        # Initialize likelihoods for each redshift bin of each survey, inheriting
        # from Likelihood_isw
        for elem in ['0','1','2','3','4']:
                exec("sdss_bin%s = type('sdss_bin%s', (Likelihood_isw, ), {})" % \
                    (elem, elem))

        self.Sdss_bin0 = sdss_bin0(
            os.path.join(self.data_directory, self.sdss_files[0]),
            data, command_line)

        self.Sdss_bin1 = sdss_bin1(
            os.path.join(self.data_directory, self.sdss_files[1]),
            data, command_line)

        self.Sdss_bin2 = sdss_bin2(
            os.path.join(self.data_directory, self.sdss_files[2]),
            data, command_line)

        self.Sdss_bin3 = sdss_bin3(
            os.path.join(self.data_directory, self.sdss_files[3]),
            data, command_line)

        self.Sdss_bin4 = sdss_bin4(
            os.path.join(self.data_directory, self.sdss_files[4]),
            data, command_line)

        for elem in ['0', '1', '2']:
                exec("qso_bin%s = type('qso_bin%s', (Likelihood_isw, ), {})" % \
                    (elem, elem))

        self.Qso_bin0 = qso_bin0(
            os.path.join(self.data_directory, self.qso_files[0]),
            data, command_line)

        self.Qso_bin1 = qso_bin1(
            os.path.join(self.data_directory, self.qso_files[1]),
            data, command_line)

        self.Qso_bin2 = qso_bin2(
            os.path.join(self.data_directory, self.qso_files[2]),
            data, command_line)

        for elem in ['0', '1', '2']:
                exec("mpz_bin%s = type('mpz_bin%s', (Likelihood_isw, ), {})" % \
                    (elem, elem))

        self.Mpz_bin0 = mpz_bin0(
            os.path.join(self.data_directory, self.mpz_files[0]),
            data, command_line)

        self.Mpz_bin1 = mpz_bin1(
            os.path.join(self.data_directory, self.mpz_files[1]),
            data, command_line)

        self.Mpz_bin2 = mpz_bin2(
            os.path.join(self.data_directory, self.mpz_files[2]),
            data, command_line)

        for elem in ['0', '1', '2']:
                exec("wisc_bin%s = type('wisc_bin%s', (Likelihood_isw, ), {})" % \
                    (elem, elem))

        self.Wisc_bin0 = wisc_bin0(
            os.path.join(self.data_directory, self.wisc_files[0]),
            data, command_line)

        self.Wisc_bin1 = wisc_bin1(
            os.path.join(self.data_directory, self.wisc_files[1]),
            data, command_line)

        self.Wisc_bin2 = wisc_bin2(
            os.path.join(self.data_directory, self.wisc_files[2]),
            data, command_line)

        for elem in ['0']:
                exec("nvss_bin%s = type('nvss_bin%s', (Likelihood_isw, ), {})" % \
                    (elem, elem))

        self.Nvss_bin0 = nvss_bin0(
            os.path.join(self.data_directory, self.nvss_files[0]),
            data, command_line)

    def loglkl(self, cosmo, data):
        # Bias parameters of each redshift bin are implemented as nuisance parameters
        b0_sdss=data.mcmc_parameters['b0_sdss']['current']*data.mcmc_parameters['b0_sdss']['scale']
        b1_sdss=data.mcmc_parameters['b1_sdss']['current']*data.mcmc_parameters['b1_sdss']['scale']
        b2_sdss=data.mcmc_parameters['b2_sdss']['current']*data.mcmc_parameters['b2_sdss']['scale']
        b3_sdss=data.mcmc_parameters['b3_sdss']['current']*data.mcmc_parameters['b3_sdss']['scale']
        b4_sdss=data.mcmc_parameters['b4_sdss']['current']*data.mcmc_parameters['b4_sdss']['scale']
        b0_qso=data.mcmc_parameters['b0_qso']['current']*data.mcmc_parameters['b0_qso']['scale']
        b1_qso=data.mcmc_parameters['b1_qso']['current']*data.mcmc_parameters['b1_qso']['scale']
        b2_qso=data.mcmc_parameters['b2_qso']['current']*data.mcmc_parameters['b2_qso']['scale']
        b0_mpz=data.mcmc_parameters['b0_mpz']['current']*data.mcmc_parameters['b0_mpz']['scale']
        b1_mpz=data.mcmc_parameters['b1_mpz']['current']*data.mcmc_parameters['b1_mpz']['scale']
        b2_mpz=data.mcmc_parameters['b2_mpz']['current']*data.mcmc_parameters['b2_mpz']['scale']
        b0_wisc=data.mcmc_parameters['b0_wisc']['current']*data.mcmc_parameters['b0_wisc']['scale']
        b1_wisc=data.mcmc_parameters['b1_wisc']['current']*data.mcmc_parameters['b1_wisc']['scale']
        b2_wisc=data.mcmc_parameters['b2_wisc']['current']*data.mcmc_parameters['b2_wisc']['scale']
        b0_nvss=data.mcmc_parameters['b0_nvss']['current']*data.mcmc_parameters['b0_nvss']['scale']
        # Compute the log-likelihood for all sublikelihoods
        loglkl=0
        loglkl+=self.Sdss_bin0.compute_loglkl(cosmo,data,b0_sdss)
        loglkl+=self.Sdss_bin1.compute_loglkl(cosmo,data,b1_sdss)
        loglkl+=self.Sdss_bin2.compute_loglkl(cosmo,data,b2_sdss)
        loglkl+=self.Sdss_bin3.compute_loglkl(cosmo,data,b3_sdss)
        loglkl+=self.Sdss_bin4.compute_loglkl(cosmo,data,b4_sdss)
        loglkl+=self.Qso_bin0.compute_loglkl(cosmo,data,b0_qso)
        loglkl+=self.Qso_bin1.compute_loglkl(cosmo,data,b1_qso)
        loglkl+=self.Qso_bin2.compute_loglkl(cosmo,data,b2_qso)
        loglkl+=self.Mpz_bin0.compute_loglkl(cosmo,data,b0_mpz)
        loglkl+=self.Mpz_bin1.compute_loglkl(cosmo,data,b1_mpz)
        loglkl+=self.Mpz_bin2.compute_loglkl(cosmo,data,b2_mpz)
        loglkl+=self.Wisc_bin0.compute_loglkl(cosmo,data,b0_wisc)
        loglkl+=self.Wisc_bin1.compute_loglkl(cosmo,data,b1_wisc)
        loglkl+=self.Wisc_bin2.compute_loglkl(cosmo,data,b2_wisc)
        loglkl+=self.Nvss_bin0.compute_loglkl(cosmo,data,b0_nvss)
        return loglkl
