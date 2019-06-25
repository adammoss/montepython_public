from argparse import Namespace
import os
from copy import copy
import warnings
import glob
import io_mp

import numpy as np
import pandas as pd

import sampler

from nnest import NestedSampler, MCMCSampler

"""
Neural network sampling
python montepython/MontePython.py run -o chains/nn_nested -p input/base2015_ns.param -m NN --NN_sampler nested --NN_n_live_points 100
python montepython/MontePython.py run -o chains/nn_mcmc -p input/base2015.param -m NN --NN_sampler mcmc 
--NN_bootstrap_fileroot chains/file
"""

NN_subfolder    = 'NN'
NN_prefix       = 'NN_'

name_paramnames = '.paramnames'
name_arguments  = '.arguments'

str2bool = lambda s: True if s.lower() == 'true' else False
NN_user_arguments = {
    # General sampling options
    'sampler':
        {'help': 'Type of sampler',
         'type': str,
         'default': 'nested'},
    'n_live_points':
        {'help': 'Number of live samples',
         'type': int,
         'default': 100},
    'switch':
        {'help': 'Switch from rejection sampling to MCMC',
         'type': float,
         'default': -1},
    'train_iters':
        {'help': 'Number of training iterations',
         'type': int,
         'default': 2000},
    'mcmc_steps':
        {'help': 'Nest sampling MCMC steps',
         'type': int,
         'default':-1},
    'fastslow':
        {'help': 'True or False',
         'type': str2bool,
         'default': True},
    # NN 
    'hidden_dim':
        {'help': 'Hidden dimension',
         'type': int,
         'default': 128},
    'hidden_layers':
        {'help': 'Number of hidden layers',
         'type': int,
         'default': 1},
    'num_blocks':
        {'help': 'Number of flow blocks',
         'type': int,
         'default': 5},
    # Ending conditions
    'evidence_tolerance':
        {'help': 'Evidence tolerance',
         'type': float,
         'default': 0.5},
    # MCMC sampler options
    'bootstrap_fileroot':
        {'help': 'Bootstrap chain fileroot',
         'type': str}
    }


def initialise(cosmo, data, command_line):

    """
    Main call to prepare the information for the NeuralNest run.
    """

    # Convenience variables
    varying_param_names = data.get_mcmc_parameters(['varying'])
    derived_param_names = data.get_mcmc_parameters(['derived'])

    if getattr(command_line, NN_prefix+'sampler', '').lower() == 'nested':

        # Check that all the priors are flat and that all the parameters are bound
        is_flat, is_bound = sampler.check_flat_bound_priors(
            data.mcmc_parameters, varying_param_names)
        if not is_flat:
            raise io_mp.ConfigurationError(
                'Nested Sampling with NeuralNest is only possible with flat ' +
                'priors. Sorry!')
        if not is_bound:
            raise io_mp.ConfigurationError(
                'Nested Sampling with NeuralNest is only possible for bound ' +
                'parameters. Set reasonable bounds for them in the ".param"' +
                'file.')

    # If absent, create the sub-folder NS
    NN_folder = os.path.join(command_line.folder, NN_subfolder)
    if not os.path.exists(NN_folder):
        os.makedirs(NN_folder)

    run_num = sum(os.path.isdir(os.path.join(NN_folder,i)) for i in os.listdir(NN_folder)) + 1

    # -- Automatic arguments
    data.NN_arguments['x_dim'] = len(varying_param_names)
    data.NN_arguments['num_derived'] =  len(derived_param_names)
    data.NN_arguments['verbose'] = True
    data.NN_arguments['log_dir'] = os.path.join(NN_folder, str(run_num))
    data.NN_arguments['use_gpu'] = False
    data.NN_arguments['flow'] = 'nvp'
    data.NN_arguments['load_model'] = ''
    data.NN_arguments['batch_size'] = 100

    if getattr(command_line, NN_prefix+'fastslow'):
        data.NN_arguments['num_slow'] = data.block_parameters[0]
    else:
        data.NN_arguments['num_slow'] = 0

    # -- User-defined arguments
    for arg in NN_user_arguments:
        value = getattr(command_line, NN_prefix+arg)
        data.NN_arguments[arg] = value
        if arg == 'switch':
            if value >= 0:
                data.NN_arguments['switch'] = value
            elif data.NN_arguments['num_slow'] > 0:
                data.NN_arguments['switch'] = 1.0 / (5 * data.NN_arguments['num_slow'])

    if getattr(command_line, NN_prefix + 'sampler', '').lower() == 'mcmc':
        data.NN_arguments['mcmc_steps'] = getattr(command_line, 'N')
            
    data.NN_param_names = varying_param_names

    base_name = os.path.join(NN_folder, 'base')

    if run_num == 1:
        # Write the NeuralNest arguments and parameter ordering
        with open(base_name+name_arguments, 'w') as afile:
            for arg in data.NN_arguments:
                afile.write(' = '.join(
                    [str(arg), str(data.NN_arguments[arg])]))
                afile.write('\n')
        with open(base_name+name_paramnames, 'w') as pfile:
            pfile.write('\n'.join(data.NN_param_names+derived_param_names))


def run(cosmo, data, command_line):

    derived_param_names = data.get_mcmc_parameters(['derived'])
    NN_param_names      = data.NN_param_names

    nDims = len(data.NN_param_names)
    nDerived = len(derived_param_names)

    if data.NN_arguments['sampler'].lower() == 'nested':

        def prior(cube):
            # NN uses cube -1 to 1 so convert to 0 to 1
            cube = cube / 2 + 0.5
            if len(cube.shape) == 1:
                theta = [0.0] * nDims
                for i, name in enumerate(data.NN_param_names):
                    theta[i] = data.mcmc_parameters[name]['prior'] \
                        .map_from_unit_interval(cube[i])
                return np.array([theta])
            else:
                thetas = []
                for c in cube:
                    theta = [0.0] * nDims
                    for i, name in enumerate(data.NN_param_names):
                        theta[i] = data.mcmc_parameters[name]['prior'] \
                            .map_from_unit_interval(c[i])
                    thetas.append(theta)
                return np.array(thetas)

        def loglike(thetas):
            logls = []
            for theta in thetas:
                try:
                    data.check_for_slow_step(theta)
                except KeyError:
                    pass
                for i, name in enumerate(data.NN_param_names):
                    data.mcmc_parameters[name]['current'] = theta[i]
                data.update_cosmo_arguments()
                # Compute likelihood
                logl = sampler.compute_lkl(cosmo, data)
                if not np.isfinite(logl):
                    print('Nan encountered in likelihood')
                    print(data.mcmc_parameters)
                logls.append(logl)
            logls = np.array(logls)
            return logls

        nn = NestedSampler(data.NN_arguments['x_dim'],
                    loglike,
                    transform=prior,
                    append_run_num=False,
                    hidden_dim=data.NN_arguments['hidden_dim'],
                    num_slow=data.NN_arguments['num_slow'],
                    num_derived=data.NN_arguments['num_derived'],
                    batch_size=data.NN_arguments['batch_size'],
                    flow=data.NN_arguments['flow'],
                    num_blocks=data.NN_arguments['num_blocks'],
                    num_layers=data.NN_arguments['hidden_layers'],
                    log_dir=data.NN_arguments['log_dir'],
                    num_live_points=data.NN_arguments['n_live_points'])

        nn.run(train_iters=data.NN_arguments['train_iters'],
               volume_switch=data.NN_arguments['switch'],
               mcmc_steps=data.NN_arguments['mcmc_steps'],
               dlogz=data.NN_arguments['evidence_tolerance'],
               mcmc_batch_size=1)

    else:

        def loglike(thetas):
            logls = []
            for theta in thetas:
                try:
                    data.check_for_slow_step(theta)
                except KeyError:
                    pass
                flag = 0
                for i, name in enumerate(data.NN_param_names):
                    value = data.mcmc_parameters[name]['initial']
                    if ((str(value[1]) != str(-1) and value[1] is not None) and
                            (theta[i] < value[1])):
                        flag += 1  # if a boundary value is reached, increment
                    elif ((str(value[2]) != str(-1) and value[2] is not None) and
                          theta[i] > value[2]):
                        flag += 1  # same
                if flag == 0:
                    for i, name in enumerate(data.NN_param_names):
                        data.mcmc_parameters[name]['current'] = theta[i]
                    data.update_cosmo_arguments()
                    # Compute likelihood
                    logl = sampler.compute_lkl(cosmo, data)
                    if not np.isfinite(logl):
                        print('Nan encountered in likelihood')
                        print(data.mcmc_parameters)
                else:
                    logl = data.boundary_loglike
                logls.append(logl)
            logls = np.array(logls)
            return logls

        nn = MCMCSampler(data.NN_arguments['x_dim'],
                         loglike,
                         append_run_num=False,
                         hidden_dim=data.NN_arguments['hidden_dim'],
                         num_slow=data.NN_arguments['num_slow'],
                         num_derived=data.NN_arguments['num_derived'],
                         batch_size=data.NN_arguments['batch_size'],
                         flow=data.NN_arguments['flow'],
                         num_blocks=data.NN_arguments['num_blocks'],
                         num_layers=data.NN_arguments['hidden_layers'],
                         log_dir=data.NN_arguments['log_dir'])

        nn.run(train_iters=data.NN_arguments['train_iters'],
               mcmc_steps=data.NN_arguments['mcmc_steps'],
               bootstrap_fileroot=data.NN_arguments['bootstrap_fileroot'],
               bootstrap_match='*__*.txt',
               bootstrap_iters=1)


def from_NN_output_to_chains(folder):
    print(folder)
    chains, logzs, nlikes = [], [], []
    for fileroot in glob.glob(folder + '/*/chains/'):
        if os.path.isfile(os.path.join(fileroot, 'chain.txt')): 
            chains.append(np.load(os.path.join(fileroot, 'chain.txt')))
        if os.path.exists(os.path.join(fileroot, '..', 'run_results', 'results.csv')):
            results = pd.read_csv(os.path.join(fileroot, '..', 'run_results', 'results.csv'))
            print(results)
        if os.path.exists(os.path.join(fileroot, '..', 'run_results', 'final.csv')):
            final = pd.read_csv(os.path.join(fileroot, '..', 'run_results', 'final.csv'))
            logzs.append(final['logz'])
            nlikes.append(final['ncall'])
    if len(logzs) > 0:
        logzs_mean = np.mean(logzs)
        nlikes_mean = np.mean(nlikes)
        if len(logzs) > 1:
            logzs_error = np.std(logzs)
            nlikes_error = np.std(nlikes)
        else:
            logzs_error = 0
            nlikes_error = 0
        print(r'Log z: %4.2f \pm %4.2f' % (logzs_mean, logzs_error))
        print(r'Number of likelihood evaluations: %.0f \pm %.0f' % (nlikes_mean, nlikes_error))
        print('')
    for ic in len(chains):
        np.savetxt(os.path.join(folder.rstrip(NN_subfolder), 'chain__%s.txt' % ic), chains[ic])
