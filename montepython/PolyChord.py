"""
.. module:: PolyChord
    :synopsis: Interface the PolyChord program with Monte Python

This implementation relies heavily on the existing Python wrapper for
PolyChord, called PyPolyChord, which comes with the PolyChord code.

The main routine, :func:`run`, truly interfaces the two codes. It takes for
input the cosmological module, data and command line. It then defines
internally two functions, :func:`prior() <PolyChord.prior>` and
:func:`loglike` that will serve as input for the run function of PyPolyChord.

.. moduleauthor:: Will Handley <wh260@cam.ac.uk>
"""
from PyPolyChord import run_polychord as polychord_run
from PyPolyChord.settings import PolyChordSettings as PC_Settings
import numpy as np
import os
from copy import copy
import warnings
import io_mp
import sampler

# Data on file names and PolyChord options, that may be called by other modules

# PolyChord subfolder and name separator
PC_subfolder    = 'PC'

name_paramnames = '.paramnames'
name_arguments  = '.arguments'
name_stats = '.stats'
name_logparam = 'log.param'

# PolyChord option prefix
PC_prefix       = 'PC_'
# User-defined arguments of PyPolyChord, and 'argparse' keywords
# First: basic string -> bool type conversion:
str2bool = lambda s: True if s.lower() == 'true' else False
PC_user_arguments = {
    'nlive':
    {'type': int,
        'help':(
        '(Default: nDims*25)\n'
        'The number of live points.\n'
        'Increasing nlive increases the accuracy of posteriors and evidences,\n'
        'and proportionally increases runtime ~ O(nlive).'
        )
        },

    'num_repeats' :
    {'type': int,
        'help':(
        '(Default: nDims*5)\n'
        'The number of slice slice-sampling steps to generate a new point.\n'
        'Increasing num_repeats increases the reliability of the algorithm.\n'
        'Typically\n'
        '* for reliable evidences need num_repeats ~ O(5*nDims).\n'
        '* for reliable posteriors need num_repeats ~ O(nDims)'
        )
        },

    'do_clustering' :
    {'type': str2bool,
        'help':(
        '(Default: True)\n'
        'Whether or not to use clustering at run time.'
        )
        },

    'feedback' :
    {'type': int,
        'help':(
        '(Default: 1)\n'
        'How much command line feedback to give\n'
        '[0,1,2,3]'
        )
        },

    'precision_criterion' :
    {'type': float,
        'help':(
        '(Default: 0.001)\n'
        'Termination criterion. Nested sampling terminates when the evidence\n'
        'contained in the live points is precision_criterion fraction of the\n'
        'total evidence.'
        )
        },

    'max_ndead' :
    {'type': int,
        'help':(
        '(Default: -1)\n'
        'Alternative termination criterion. Stop after max_ndead iterations.\n'
        'Set negative to ignore (default).'
        )
        },

    'boost_posterior' :
        {'type': float,
                'help':(
        '(Default: 0.0)\n'
        'Increase the number of posterior samples produced.  This can be set\n'
        'arbitrarily high, but you won\'t be able to boost by more than\n'
        'num_repeats\n'
        'Warning: in high dimensions PolyChord produces _a lot_ of posterior\n'
        'samples. You probably don\'t need to change this'
        )
        },

    'posteriors' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Produce (weighted) posterior samples. Stored in <root>.txt.'
        )
        },

    'equals' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Produce (equally weighted) posterior samples. Stored in\n'
        '<root>_equal_weights.txt'
        )
        },

    'cluster_posteriors' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Produce posterior files for each cluster?\n'
        'Does nothing if do_clustering=False.'
        )
        },

    'write_resume' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Create a resume file.'
        )
        },

    'read_resume' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Read from resume file.'
        )
        },

    'write_stats' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Write an evidence statistics file.'
        )
        },

    'write_live' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Write a live points file.'
        )
        },

    'write_dead' :
        {'type': str2bool,
                'help':(
        '(Default: True)\n'
        'Write a dead points file.'
        )
        },

    'compression_factor' :
        {'type': double,
                'help':(
        '(Default: exp(-1))\n'
        'How often to update the files and do clustering.'
        )
        }
    }


# Automatically-defined arguments of PyMultiNest, type specified
PC_auto_arguments = {
    'file_root': {'type': str},
    'base_dir': {'type': str},
    'grade_dims': {'type': list},
    'grade_frac': {'type': list}
    }


def initialise(cosmo, data, command_line):
    """
    Main call to prepare the information for the MultiNest run.
    """

    # Convenience variables
    varying_param_names = data.get_mcmc_parameters(['varying'])
    derived_param_names = data.get_mcmc_parameters(['derived'])
    nslow = len(data.get_mcmc_parameters(['varying', 'cosmo']))
    nfast = len(data.get_mcmc_parameters(['varying', 'nuisance']))

    # Check that all the priors are flat and that all the parameters are bound
    is_flat, is_bound = sampler.check_flat_bound_priors(
        data.mcmc_parameters, varying_param_names)
    if not is_flat:
        raise io_mp.ConfigurationError(
            'Nested Sampling with PolyChord is only possible ' +
            'with flat priors. Sorry!')
    if not is_bound:
        raise io_mp.ConfigurationError(
            'Nested Sampling with PolyChord is only possible ' +
            'for bound parameters. Set reasonable bounds for them in the ' +
            '".param" file.')

    # If absent, create the sub-folder PC
    PC_folder = os.path.join(command_line.folder, PC_subfolder)
    if not os.path.exists(PC_folder):
        os.makedirs(PC_folder)

    # If absent, create the sub-folder PC/clusters
    PC_clusters_folder = os.path.join(PC_folder,'clusters') 
    if not os.path.exists(PC_clusters_folder):
        os.makedirs(PC_clusters_folder)

    # Use chain name as a base name for PolyChord files
    chain_name = [a for a in command_line.folder.split(os.path.sep) if a][-1]
    base_name = os.path.join(PC_folder, chain_name)

    # Prepare arguments for PyPolyChord
    # -- Automatic arguments
    data.PC_arguments['file_root'] = chain_name
    data.PC_arguments['base_dir'] = PC_folder
    data.PC_arguments['grade_dims'] = [nslow, nfast]
    data.PC_arguments['grade_frac'] = [0.75,0.25]
    data.PC_arguments['num_repeats'] = nslow * 2

    # -- User-defined arguments
    for arg in PC_user_arguments:
        value = getattr(command_line, PC_prefix+arg)
        if value != -1:
            data.PC_arguments[arg] = value
        # else: don't define them -> use PyPolyChord default value

    data.PC_param_names = varying_param_names

    # Write the PolyChord arguments and parameter ordering
    with open(base_name+name_arguments, 'w') as afile:
        for arg in data.PC_arguments:
            afile.write(' = '.join(
                [str(arg), str(data.PC_arguments[arg])]))
            afile.write('\n')
    with open(base_name+name_paramnames, 'w') as pfile:
        pfile.write('\n'.join(data.PC_param_names+derived_param_names))


def run(cosmo, data, command_line):
    """
    Main call to run the PolyChord sampler.

    Note the unusual set-up here, with the two following functions, `prior` and
    `loglike` having their docstrings written in the encompassing function.
    This trick was necessary as PolyChord required these two functions to be
    defined with a given number of parameters, so we could not add `data`. By
    defining them inside the run function, this problem was by-passed.

    .. function:: prior

        Generate the prior function for PolyChord

        It should transform the input unit cube into the parameter cube. This
        function actually wraps the method :func:`map_from_unit_interval()
        <prior.Prior.map_from_unit_interval>` of the class :class:`Prior
        <prior.Prior>`.

        Parameters
        ----------
        cube : list
            Contains the current point in unit parameter space that has been
            selected within the PolyChord part.
        Returns
        -------
        theta : list
            The transformed physical parameters


    .. function:: loglike

        Generate the Likelihood function for PolyChord

        Parameters
        ----------
        theta : array
            Contains the current point in the correct parameter space after
            transformation from :func:`prior`.
        Returns
        -------
        logl : float
            The loglikelihood of theta
        phi : list
            The derived parameters


    """
    # Convenience variables
    derived_param_names = data.get_mcmc_parameters(['derived'])
    nDims = len(data.PC_param_names)
    nDerived = len(derived_param_names)

    # Function giving the prior probability
    def prior(hypercube):
        """
        Please see the encompassing function docstring

        """
        theta = [0.0] * nDims
        for i, name in enumerate(data.PC_param_names):
            theta[i] = data.mcmc_parameters[name]['prior']\
                .map_from_unit_interval(hypercube[i])
        return theta

    # Function giving the likelihood probability
    def loglike(theta):
        """
        Please see the encompassing function docstring

        """
        # Updates values: theta --> data
        try:
            data.check_for_slow_step(theta)
        except KeyError:
            pass

        for i, name in enumerate(data.PC_param_names):
            data.mcmc_parameters[name]['current'] = theta[i]
        data.update_cosmo_arguments()

        # Compute likelihood
        logl = sampler.compute_lkl(cosmo, data)

        # Compute derived parameters and pass them back
        phi = [0.0] * nDerived
        for i, name in enumerate(derived_param_names):
            phi[i] = float(data.mcmc_parameters[name]['current'])

        return logl, phi

    # Pass over the settings
    settings = PC_Settings(nDims,nDerived)
    for arg, val in data.PC_arguments.iteritems():
        setattr(settings, arg, val)

    # Launch PolyChord
    polychord_run(loglike, nDims, nDerived, settings, prior)

    warnings.warn('The sampling with PolyChord is done.\n' +
                  'You can now analyse the output calling Monte Python ' +
                  ' with the -info flag in the chain_name/PC subfolder,')
