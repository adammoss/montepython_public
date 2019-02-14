# there is no specific likelihood code for this experiment, because it
# falls in the category of CMB experiments described in the "mock CMB"
# format. The class below inherits the properties of a general class
# "Likelihood_mock_cmb", which knows how to deal with all experiments in
# "mock CMB" format.

from montepython.likelihood_class import Likelihood_mock_cmb


class litebird_highl_fsky0p3(Likelihood_mock_cmb):
    pass
