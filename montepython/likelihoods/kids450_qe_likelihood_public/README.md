This repository contains the likelihood module for the KiDS-450 shear power spectrum measurements (derived using a quadratic estimator) from [Köhlinger et al. 2017 (MNRAS, 471, 4412)](http://adsabs.harvard.edu/abs/2017MNRAS.471.4412K). The module will be working 'out-of-the-box' within a [MontePython](https://github.com/baudren/montepython_public) and [CLASS](https://github.com/lesgourg/class_public) (version >= 2.6!) setup. The required data files can be downloaded from 'http://kids.strw.leidenuniv.nl/sciencedata.php' and the parameter files for reproducing the fiducial results of the paper are supplied in the subfolder 'input' within this repository. 

Assuming that MontePython (with CLASS version >= 2.6) is set up (we recommend to use the MultiNest sampler!), please proceed as follows:

1) Clone this repository

`git clone https://bitbucket.org/fkoehlin/kids450_qe_likelihood_public.git`

2) Copy `__init__.py` and `kids450_qe_likelihood_public.data` from this repository into a folder named `kids450_qe_likelihood_public` within `/your/path/to/montepython_public/montepython/likelihoods/`.

(you can rename the folder to whatever you like, but you must use this name then consistently for the whole likelihood which implies to rename the `*.data`-file, including the prefixes of the parameters defined in there, the name of the likelihood in the `__init__.py`-file and also in the `*.param`-file.)

3) Set the path to the data folder (`data_for_likelihood` from the tarball available at the KiDS webpage listed above) in `kids450_qe_likelihood_public.data` and modify parameters as you please (note that everything is set up to repeat the fiducial 3 z-bin analysis with `fiducial_3zbins.params`).

3) Start your runs using e.g. the `fiducial_<n>zbins.params` (<n>=2, 3) supplied in the subfolder `input` within this repository.

4) Contribute your developments/bugfixes to this likelihood (please use a dedicated branch per fix/feature).

5) If you publish your results based on using this likelihood (and data), please cite [Köhlinger et al. 2017 (MNRAS, 471, 4412)](http://adsabs.harvard.edu/abs/2017MNRAS.471.4412K) and all relevant references for MontePython and CLASS.

Refer to `run_fiducial_with_multinest.sh` within the subfolder `input` for all MultiNest-related settings that were used for the fiducial runs.

For questions/comments please use the issue-tracking system!
