"""
.. module:: sampler
    :synopsis: Generic sampler
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
.. moduleauthor:: Surhudm More <>

This module defines one key function, :func:`run`, that distributes the work to
the desired actual sampler (Metropolis Hastings, or Nested Sampling so far).

It also defines a serie of helper functions, that aim to be generically used by
all different sampler methods:

* :func:`get_covariance_matrix`
* :func:`read_args_from_chain`
* :func:`read_args_from_bestfit`
* :func:`accept_step`
* :func:`compute_lkl`


"""
import numpy as np
import sys
import warnings

import io_mp
import os
import scipy.linalg as la
import scipy.optimize as op

def run(cosmo, data, command_line):
    """
    Depending on the choice of sampler, dispatch the appropriate information

    The :mod:`mcmc` module is used as previously, except the call to
    :func:`mcmc.chain`, or :func:`nested_sampling.run` is now within
    this function, instead of from within :mod:`MontePython`.

    In the long term, this function should contain any potential hybrid scheme.

    """

    if command_line.method == 'MH':
        import mcmc
        mcmc.chain(cosmo, data, command_line)
        data.out.close()
    elif command_line.method == 'NS':
        import nested_sampling as ns
        ns.run(cosmo, data, command_line)
    elif command_line.method == 'CH':
        import cosmo_hammer as hammer
        hammer.run(cosmo, data, command_line)
    elif command_line.method == 'IS':
        import importance_sampling as ims
        ims.run(cosmo, data, command_line)
    elif command_line.method == 'Der':
        import add_derived as der
        der.run(cosmo, data, command_line)
    else:
        raise io_mp.ConfigurationError(
            "Sampling method %s not understood" % command_line.method)


def read_args_from_chain(data, chain):
    """
    Pick up the last accepted values from an input chain as a starting point

    Function used only when the restart flag is set. It will simply read the
    last line of an input chain, using the tail command from the extended
    :class:`io_mp.File` class.

    .. warning::
        That method was not tested since the adding of derived parameters. The
        method :func:`read_args_from_bestfit` is the prefered one.

    .. warning::
        This method works because of the particular presentation of the chain,
        and the use of tabbings (not spaces). Please keep this in mind if you
        are having difficulties

    Parameters
    ----------
    chain : str
        Name of the input chain provided with the command line.

    """
    chain_file = io_mp.File(chain, 'r')
    parameter_names = data.get_mcmc_parameters(['varying'])

    i = 1
    for elem in parameter_names:
        data.mcmc_parameters[elem]['last_accepted'] = float(
            chain_file.tail(1)[0].split('\t')[i])
        i += 1


def read_args_from_bestfit(data, bestfit):
    """
    Deduce the starting point either from the input file, or from a best fit
    file.

    Parameters
    ----------
    bestfit : str
        Name of the bestfit file from the command line.

    """

    parameter_names = data.get_mcmc_parameters(['varying'])
    bestfit_file = open(bestfit, 'r')
    for line in bestfit_file:
        if line.find('#') != -1:
            bestfit_names = line.strip('#').replace(' ', '').\
                replace('\n', '').split(',')
            bestfit_values = np.zeros(len(bestfit_names), 'float64')
        else:
            line = line.split()
            for i in range(len(line)):
                bestfit_values[i] = line[i]

    print
    print('\nStarting point for rescaled parameters:')
    for elem in parameter_names:
        if elem in bestfit_names:
            data.mcmc_parameters[elem]['last_accepted'] = \
                bestfit_values[bestfit_names.index(elem)] / \
                data.mcmc_parameters[elem]['scale']
            print 'from best-fit file : ', elem, ' = ',
            print bestfit_values[bestfit_names.index(elem)] / \
                data.mcmc_parameters[elem]['scale']
        else:
            data.mcmc_parameters[elem]['last_accepted'] = \
                data.mcmc_parameters[elem]['initial'][0]
            print 'from input file    : ', elem, ' = ',
            print data.mcmc_parameters[elem]['initial'][0]


def get_covariance_matrix(cosmo, data, command_line):
    """
    Compute the covariance matrix, from an input file or from an existing
    matrix.

    Reordering of the names and scaling take place here, in a serie of
    potentially hard to read methods. For the sake of clarity, and to avoid
    confusions, the code will, by default, print out a succession of 4
    covariance matrices at the beginning of the run, if starting from an
    existing one. This way, you can control that the paramters are set
    properly.

    .. note::

        The set of parameters from the run need not to be the exact same
        set of parameters from the existing covariance matrix (not even the
        ordering). Missing parameter from the existing covariance matrix will
        use the sigma given as an input.

    """

    # Setting numpy options in terms of precision (useful when writing to files
    # or displaying a result, but does not affect the precision of the
    # computation).
    np.set_printoptions(precision=2, linewidth=150)
    parameter_names = data.get_mcmc_parameters(['varying'])

    # Define quiet setting if not previously defined
    try:
        command_line.quiet
    except:
        command_line.quiet = False

    # if the user provides a .covmat file or if user asks to compute a fisher matrix
    if command_line.cov is not None:

        cov = open('{0}'.format(command_line.cov), 'r')

        i = 0
        for line in cov:
            if line.find('#') != -1:
                # Extract the names from the first line
                covnames = line.strip('#').replace(' ', '').\
                    replace('\n', '').split(',')
                # Initialize the matrices
                matrix = np.zeros((len(covnames), len(covnames)), 'float64')
                rot = np.zeros((len(covnames), len(covnames)))
            else:
                line = line.split()
                for j in range(len(line)):
                    matrix[i][j] = np.array(line[j], 'float64')
                i += 1

        # First print out
        if not command_line.silent and not command_line.quiet:
            print('\nInput covariance matrix:')
            print(covnames)
            print(matrix)
        # Deal with the all problematic cases.
        # First, adjust the scales between stored parameters and the ones used
        # in mcmc
        scales = []
        for elem in covnames:
            if elem in parameter_names:
                scales.append(data.mcmc_parameters[elem]['scale'])
            else:
                scales.append(1)
        scales = np.diag(scales)
        # Compute the inverse matrix, and assert that the computation was
        # precise enough, by comparing the product to the identity matrix.
        invscales = np.linalg.inv(scales)
        np.testing.assert_array_almost_equal(
            np.dot(scales, invscales), np.eye(np.shape(scales)[0]),
            decimal=5)

        # Apply the newly computed scales to the input matrix
        matrix = np.dot(invscales.T, np.dot(matrix, invscales))

        # Second print out, after having applied the scale factors
        if not command_line.silent and not command_line.quiet:
            print('\nFirst treatment (scaling)')
            print(covnames)
            print(matrix)

        # Rotate matrix for the parameters to be well ordered, even if some
        # names are missing or some are in extra.
        # First, store the parameter names in temp_names that also appear in
        # the covariance matrix, in the right ordering for the code (might be
        # different from the input matri)
        temp_names = [elem for elem in parameter_names if elem in covnames]

        # If parameter_names contains less things than covnames, we will do a
        # small trick. Create a second temporary array, temp_names_2, that will
        # have the same dimension as covnames, and containing:
        # - the elements of temp_names, in the order of parameter_names (h
        # index)
        # - an empty string '' for the remaining unused parameters
        temp_names_2 = []
        h = 0
        not_in = [elem for elem in covnames if elem not in temp_names]
        for k in range(len(covnames)):
            if covnames[k] not in not_in:
                temp_names_2.append(temp_names[h])
                h += 1
            else:
                temp_names_2.append('')

        # Create the rotation matrix, that will put the covariance matrix in
        # the right order, and also assign zeros to the unused parameters from
        # the input. These empty columns will be removed in the next step.
        for k in range(len(covnames)):
            for h in range(len(covnames)):
                try:
                    if covnames[k] == temp_names_2[h]:
                        rot[h][k] = 1.
                    else:
                        rot[h][k] = 0.
                except IndexError:
                    # The IndexError exception means that we are dealing with
                    # an unused parameter. By enforcing the corresponding
                    # rotation matrix element to 0, the resulting matrix will
                    # still have the same size as the original, but with zeros
                    # on the unused lines.
                    rot[h][k] = 0.
        matrix = np.dot(rot, np.dot(matrix, np.transpose(rot)))

        # Third print out
        if not command_line.silent and not command_line.quiet:
            print('\nSecond treatment (partial reordering and cleaning)')
            print(temp_names_2)
            print(matrix)

        # Final step, creating a temporary matrix, filled with 1, that will
        # eventually contain the result.
        matrix_temp = np.ones((len(parameter_names),
                               len(parameter_names)), 'float64')
        indices_final = np.zeros(len(parameter_names))
        indices_initial = np.zeros(len(covnames))
        # Remove names that are in parameter names but not in covnames, and
        # set to zero the corresponding columns of the final result.
        for k in range(len(parameter_names)):
            if parameter_names[k] in covnames:
                indices_final[k] = 1
        for zeros in np.where(indices_final == 0)[0]:
            matrix_temp[zeros, :] = 0
            matrix_temp[:, zeros] = 0
        # Remove names that are in covnames but not in param_names
        for h in range(len(covnames)):
            if covnames[h] in parameter_names:
                indices_initial[h] = 1
        # There, put a place holder number (we are using a pure imaginary
        # number: i, to avoid any problem) in the initial matrix, so that the
        # next step only copy the interesting part of the input to the final
        # matrix.
        max_value = np.finfo(np.float64).max
        for zeros in np.where(indices_initial == 0)[0]:
            matrix[zeros, :] = [max_value for _ in range(
                len(matrix[zeros, :]))]
            matrix[:, zeros] = [max_value for _ in range(
                len(matrix[:, zeros]))]
        # Now put in the temporary matrix, where the 1 were, the interesting
        # quantities from the input (the one that are not equal to i).
        matrix_temp[matrix_temp == 1] = matrix[matrix != max_value]
        matrix = np.copy(matrix_temp)
        # on all other lines, that contain 0, just use sigma^2
        for zeros in np.where(indices_final == 0)[0]:
            matrix[zeros, zeros] = np.array(
                data.mcmc_parameters[parameter_names[zeros]]['initial'][3],
                'float64')**2
    # else, take sigmas^2.
    else:
        matrix = np.identity(len(parameter_names), 'float64')
        for index, elem in enumerate(parameter_names):
            matrix[index][index] = np.array(
                data.mcmc_parameters[elem]['initial'][3], 'float64')**2


    # Final print out, the actually used covariance matrix
    if not command_line.silent and not command_line.quiet:
        sys.stdout.write('\nDeduced starting covariance matrix:\n')
        print(parameter_names)
        print(matrix)

    #inverse, and diagonalization
    eigv, eigV = np.linalg.eig(np.linalg.inv(matrix))

    #if command_line.start_from_fisher:
    #    command_line.fisher = True
    #if command_line.fisher:
    #    eigv, eigV, matrix = get_fisher_matrix(cosmo, data, command_line, matrix)

    return eigv, eigV, matrix

def get_minimum(cosmo, data, command_line, covmat):

    if not command_line.silent:
        warnings.warn("Minimization implementation is being tested")

    # Create the center dictionary, which will hold the center point
    # information
    center = {}
    parameter_names = data.get_mcmc_parameters(['varying'])

    if not command_line.bf:
        for elem in parameter_names:
            #temp_data.mcmc_parameters[elem]['current'] = (
            #    data.mcmc_parameters[elem]['initial'][0])
            center[elem] = data.mcmc_parameters[elem]['initial'][0]
    else:
        #read_args_from_bestfit(temp_data, command_line.bf)
        read_args_from_bestfit(data, command_line.bf)
        for elem in parameter_names:
            #temp_data.mcmc_parameters[elem]['current'] = (
            #    temp_data.mcmc_parameters[elem]['last_accepted'])
            #center[elem] = temp_data.mcmc_parameters[elem]['last_accepted']
            center[elem] = data.mcmc_parameters[elem]['last_accepted']

    print center
    #print chi2_eff(center, cosmo, data)
    #print gradient_chi2_eff(center, cosmo, data)

    stepsizes = np.zeros(len(parameter_names), 'float64')
    parameters = np.zeros(len(parameter_names), 'float64')
    bounds = np.zeros([len(parameter_names),2], 'float64')
    cons = ()
    for index, elem in enumerate(parameter_names):
        parameters[index] = center[elem]
        #stepsizes[index] = center[elem]*0.0001
        stepsizes[index] = 0.1*covmat[index,index]**0.5
        bounds[index,0] = center[elem] - 1*covmat[index,index]**0.5
        bounds[index,1] = center[elem] + 1*covmat[index,index]**0.5
        cons += ({'type': 'ineq', 'fun': lambda x: x[index] - bounds[index,0]},
                 {'type': 'ineq', 'fun': lambda x: bounds[index,1] - x[index]},)
        print bounds[index,0],bounds[index,1]

    print parameters
    print stepsizes

    #minimum, chi2 = op.fmin_cg(chi2_eff,
    # Use unconstrained Polak & Ribiere conjugate gradient algorithm
    # CosmoMC uses a constrained (Fletcher & Reeves) version of this 
    #xopt, fopt, func_calls, grad_calls, warnflags, allvecs = op.fmin_cg(chi2_eff,
    #                                                                    parameters,
    #                                                                    #fprime = gradient_chi2_eff,
    #                                                                    epsilon = stepsizes,
    #                                                                    args = (cosmo,data),#bounds),
    #                                                                    full_output = True,
    #                                                                    disp = True,
    #                                                                    retall = True)
    # Use constrained Newton conjugate gradient algorithm
    #x, nfeval, rc = op.fmin_tnc(chi2_eff,
    #                            parameters,
    #                            #fprime = gradient_chi2_eff,
    #                            args = (cosmo,data),
    #                            approx_grad = True,
    #                            bounds = bounds,
    #                            epsilon = stepsizes,
    #                            disp = 5)

    #result = op.minimize(chi2_eff,
    #                     parameters,
    #                     args = (cosmo,data),
    #                     method='COBYLA',
    #                     #method='SLSQP',
    #                     constraints=cons,
    #                     #bounds=bounds,
    #                     tol=0.000001,
    #                     options = {'disp': True,
    #                                'rhobeg': stepsizes})
    #                                #'eps': stepsizes})

    result = op.minimize(chi2_eff,
                         parameters,
                         args = (cosmo,data),
                         #method='trust-region-exact',
                         method='BFGS',
                         #constraints=cons,
                         #bounds=bounds,
                         tol=0.0001,
                         options = {'disp': True})
                                    #'initial_tr_radius': stepsizes,
                                    #'max_tr_radius': stepsizes})


    ##print minimum
    ##print chi2
    ##print success
    #for index,elem in enumerate(parameter_names):
    #    print elem, xopt[i]
    #print 'chisq', fopt
    #print 'funcion calls', func_calls
    #print 'gradient calls', grad_calls
    #print 'warning flags',warnflags
    ##print allvecs
    #for index,elem in enumerate(parameter_names):
    #    print elem, x[i]
    ##print x
    #print 'nfeval',nfeval
    #print 'rc',rc
    print result
    exit()

    return center

def chi2_eff(params, cosmo, data, bounds=False):
    parameter_names = data.get_mcmc_parameters(['varying'])
    for index, elem in enumerate(parameter_names):
        #print elem,params[index]
        data.mcmc_parameters[elem]['current'] = params[index]
        if not type(bounds) == type(False):
            if (params[index] < bounds[index,0]) or (params[index] > bounds[index,1]):
                chi2 = 1e30
                print elem+' exceeds bounds with value %f and bounds %f < x < %f' %(params[index],bounds[index,0],bounds[index,1])
                return chi2
    # Update current parameters to the new parameters, only taking steps as requested
    data.update_cosmo_arguments()
    # Compute loglike value for the new parameters
    chi2 = -2.*compute_lkl(cosmo, data)
    print chi2,' at ',params
    return chi2

def gradient_chi2_eff(params, cosmo, data, bounds=False):
    parameter_names = data.get_mcmc_parameters(['varying'])
    for index, elem in enumerate(parameter_names):
        data.mcmc_parameters[elem]['current'] = params[index]
    # Update current parameters to the new parameters, only taking steps as requested
    data.update_cosmo_arguments()
    # Compute loglike value for the new parameters
    chi2 = -2.*compute_lkl(cosmo, data)
    # Initialise the gradient field
    gradient = np.zeros(len(parameter_names), 'float64')
    for index, elem in enumerate(parameter_names):
        dx = 0.01*params[index]
        #

        data.mcmc_parameters[elem]['current'] += dx
        data.update_cosmo_arguments()
        chi2_plus = -2.*compute_lkl(cosmo, data)
        #
        data.mcmc_parameters[elem]['current'] -= 2.*dx
        data.update_cosmo_arguments()
        chi2_minus = -2.*compute_lkl(cosmo, data)
        #
        gradient[index] = (chi2_plus - chi2_minus)/2./dx
        #
        data.mcmc_parameters[elem]['current'] += dx
    return gradient

def get_fisher_matrix(cosmo, data, command_line, inv_fisher_matrix):
    # Adapted by T. Brinckmann, T. Tram
    # We will work out the fisher matrix for all the parameters and
    # write it to a file

    # Pass input to data structure
    data.fisher_delta = command_line.fisher_delta
    data.fisher_tol = command_line.fisher_tol

    # Fisher method options (1,2, and 3 are experimental)
    # fisher_mode=3 use covariance matrix in 2-d for off-diagonal elements
    # fisher_mode=2 use Cholesky decomposition to rotate parameter space
    # fisher_mode=1 use eigenvectors of covariance matrix to rotate parameter space
    # fisher_mode=0 use non-rotated parameter space (recommended)
    data.fisher_mode = command_line.fisher_mode
    print 'delta, tol, mode =', data.fisher_delta, data.fisher_tol, command_line.fisher_mode
    if data.fisher_mode not in [0,1,2,3]:
        raise io_mp.ConfigurationError("Fisher mode not valid. Available modes: 0 (standard), 1 (eigevenctor rotation), "
                                       "2 (Cholesky rotation), 3 (2d rotation for off-diagonal elements). "
                                       "Modes 1, 2 and 3 are experimental.")

    # Additional option relevant for fisher_mode=2
    # use_cholesky_step=True use Cholesky decomposition to determine stepsize
    # use_cholesky_step=False use input stepsize from covariance matrix of param file
    data.use_cholesky_step = False
    if data.fisher_mode == 2:
        data.use_cholesky_step = True
    # Force step to always be symmetric (recommended)
    data.use_symmetric_step = True
    # Rotate back to cosmological parameter basis
    data.rotate_back = False

    if not command_line.silent:
        warnings.warn("Fisher implementation is being tested")

    # Let us create a separate copy of data
    #from copy import deepcopy
    # Do not modify data, instead copy
    #temp_data = deepcopy(data)
    #done = False

    # Create the center dictionary, which will hold the center point
    # information
    center = {}
    parameter_names = data.get_mcmc_parameters(['varying'])
    if not command_line.bf:
        for elem in parameter_names:
            #temp_data.mcmc_parameters[elem]['current'] = (
            #    data.mcmc_parameters[elem]['initial'][0])
            center[elem] = data.mcmc_parameters[elem]['initial'][0]
    else:
        #read_args_from_bestfit(temp_data, command_line.bf)
        read_args_from_bestfit(data, command_line.bf)
        for elem in parameter_names:
            #temp_data.mcmc_parameters[elem]['current'] = (
            #    temp_data.mcmc_parameters[elem]['last_accepted'])
            #center[elem] = temp_data.mcmc_parameters[elem]['last_accepted']
            center[elem] = data.mcmc_parameters[elem]['last_accepted']

    scales = np.zeros((len(parameter_names)))
    invscales = np.zeros((len(parameter_names)))
    for index, elem in enumerate(parameter_names):
        data.mcmc_parameters[elem]['current'] = center[elem]
        scales[index] = data.mcmc_parameters[elem]['scale']
        invscales[index] = 1./data.mcmc_parameters[elem]['scale']

    # Load stepsize from input covmat or covmat generated from param file
    # JL TODO: check this, and try another scheme to be sure that index and elem refer to the same params in the same order
    # here the stepsizes are for the scaled parameter (e.g. 100*omega_b)
    stepsize = np.zeros([len(parameter_names),3])
    for index in range(len(parameter_names)):
        stepsize[index,0] = -(inv_fisher_matrix[index][index])**0.5
        stepsize[index,1] = (inv_fisher_matrix[index][index])**0.5
    # Adjust stepsize in case step exceeds boundary
    stepsize = adjust_fisher_bounds(data,center,stepsize)

    fisher_iteration = 0
    while fisher_iteration < command_line.fisher_it:
        fisher_iteration += 1
        # Compute the Fisher matrix and the gradient array at the center
        # point.
        print ("Compute Fisher [iteration %d/%d] with following stepsizes for scaled parameters:" % (fisher_iteration,command_line.fisher_it))
        for index in range(len(parameter_names)):
            #print "%s : left %e, right %e" % (parameter_names[index],stepsize[index,0],stepsize[index,1])
            print "%s : diagonal element = %e" % (parameter_names[index],inv_fisher_matrix[index,index])

        if data.fisher_mode == 1:
            # Compute eigenvectors in order to take steps in the basis
            # of the covariance matrix instead of the basis of the parameter space
            sigma_eig, step_matrix = la.eig(inv_fisher_matrix)
        elif data.fisher_mode == 2:
            step_matrix = la.cholesky(inv_fisher_matrix).T
        elif data.fisher_mode == 3:
            step_matrix = inv_fisher_matrix
        else:
            step_matrix = np.identity(len(parameter_names), dtype='float64')

        # Compute fisher matrix
        fisher_matrix, gradient = compute_fisher(data, cosmo, center, stepsize, step_matrix)
        # If we want to rotate back to the cosmological parameter basis
        if (data.fisher_mode == 1 or data.fisher_mode == 2) and data.rotate_back:
            fisher_matrix = step_matrix.T * fisher_matrix * step_matrix
        if not command_line.silent:
            print ("Fisher matrix computed [iteration %d/%d]" % (fisher_iteration,command_line.fisher_it))

        # Compute inverse of the fisher matrix, catch LinAlgError exception
        try:
            inv_fisher_matrix = np.linalg.inv(fisher_matrix)
        except np.linalg.LinAlgError:
            raise io_mp.ConfigurationError(
                "Could not find Fisher matrix inverse, please adjust bestfit and/or input "
                "sigma values (or covmat) or remove the option --fisher and run "
                "with Metropolis-Hastings or another sampling method.")

        # Check if the inverse fisher matrix is positive definite
        fisher_status = 1
        try:
            la.cholesky(inv_fisher_matrix).T
        except:
            print 'Fisher matrix computation failed - inverse not positive definite'
            print 'Increasing fisher_it by 1, to %d, and adjusting fisher_tol and fisher_delta' %(command_line.fisher_it + 1)
            fisher_status = 0
            command_line.fisher_it += 1
            if fisher_iteration == 20:
                print 'Fisher matrix not obtained after 20 attempts, aborting.'
                print 'Try with different fisher_tol, fisher_delta, input sigmas or covmat'
            if data.fisher_tol < 0.25 * command_line.fisher_tol:
                print 'Resetting fisher_tol to %f and changing fisher_delta from %f to %f' %(command_line.fisher_tol,data.fisher_delta,0.5*data.fisher_delta)
                data.fisher_tol = command_line.fisher_tol
                data.fisher_delta *= 0.5
            else:
                print 'Reducing fisher_tol from %f to %f for fisher_delta %f' %(data.fisher_tol,0.5*data.fisher_tol,data.fisher_delta)
                data.fisher_tol *= 0.5

        # Update stepsize for the next iteration after successful fisher matrix computation
        if fisher_status:
            stepsize = np.zeros([len(parameter_names),3])
            for index in range(len(parameter_names)):
                stepsize[index,0] = -(inv_fisher_matrix[index,index])**0.5
                stepsize[index,1] = (inv_fisher_matrix[index,index])**0.5
            # Adjust stepsize in case step exceeds boundary
            stepsize = adjust_fisher_bounds(data,center,stepsize)

        # Take scalings into account and write the matrices in files
        fisher_matrix = invscales[:,np.newaxis]*fisher_matrix*invscales[np.newaxis,:]
        io_mp.write_covariance_matrix(
            fisher_matrix, parameter_names,
            os.path.join(command_line.folder, 'fisher'+str(fisher_iteration)+'.mat'))

        inv_fisher_matrix = scales[:,np.newaxis]*inv_fisher_matrix*scales[np.newaxis,:]
        io_mp.write_covariance_matrix(
            inv_fisher_matrix, parameter_names,
            os.path.join(command_line.folder, 'inv_fisher'+str(fisher_iteration)+'.mat'))

    # Removing scale factors in order to store true parameter covariance
    #inv_fisher_matrix = scales[:,np.newaxis]*inv_fisher_matrix*scales[np.newaxis,:]

    # Write the last inverse Fisher matrix as the new covariance matrix
    io_mp.write_covariance_matrix(
        inv_fisher_matrix, parameter_names,
        os.path.join(command_line.folder, 'covariance_fisher.covmat'))

    # Load the covmat from computed fisher matrix as the new starting covariance matrix
    # eigv, eigV, matrix = get_covariance_matrix(cosmo, data, command_line)

    return inv_fisher_matrix


def accept_step(data):
    """
    Transfer the 'current' point in the varying parameters to the last accepted
    one.

    """
    for elem in data.get_mcmc_parameters(['varying']):
        data.mcmc_parameters[elem]['last_accepted'] = \
            data.mcmc_parameters[elem]['current']
    for elem in data.get_mcmc_parameters(['derived']):
        data.mcmc_parameters[elem]['last_accepted'] = \
            data.mcmc_parameters[elem]['current']


def check_flat_bound_priors(parameters, names):
    """
    Ensure that all varying parameters are bound and flat

    It is a necessary condition to use the code with Nested Sampling or the
    Cosmo Hammer.
    """
    is_flat = all(parameters[name]['prior'].prior_type == 'flat'
                  for name in names)
    is_bound = all(parameters[name]['prior'].is_bound()
                   for name in names)
    return is_flat, is_bound


def compute_lkl(cosmo, data):
    """
    Compute the likelihood, given the current point in parameter space.

    This function now performs a test before calling the cosmological model
    (**new in version 1.2**). If any cosmological parameter changed, the flag
    :code:`data.need_cosmo_update` will be set to :code:`True`, from the
    routine :func:`check_for_slow_step <data.Data.check_for_slow_step>`.

    Returns
    -------
    loglike : float
        The log of the likelihood (:math:`\\frac{-\chi^2}2`) computed from the
        sum of the likelihoods of the experiments specified in the input
        parameter file.

        This function returns :attr:`data.boundary_loglike
        <data.data.boundary_loglike>`, defined in the module :mod:`data` if
        *i)* the current point in the parameter space has hit a prior edge, or
        *ii)* the cosmological module failed to compute the model. This value
        is chosen to be extremly small (large negative value), so that the step
        will always be rejected.


    """
    from classy import CosmoSevereError, CosmoComputationError

    # If the cosmological module has already been called once, and if the
    # cosmological parameters have changed, then clean up, and compute.
    if cosmo.state and data.need_cosmo_update is True:
        cosmo.struct_cleanup()

    # If the data needs to change, then do a normal call to the cosmological
    # compute function. Note that, even if need_cosmo update is True, this
    # function must be called if the jumping factor is set to zero. Indeed,
    # this means the code is called for only one point, to set the fiducial
    # model.
    if ((data.need_cosmo_update) or
            (not cosmo.state) or
            (data.jumping_factor == 0)):

        # Prepare the cosmological module with the new set of parameters
        cosmo.set(data.cosmo_arguments)

        # Compute the model, keeping track of the errors

        # In classy.pyx, we made use of two type of python errors, to handle
        # two different situations.
        # - CosmoSevereError is returned if a parameter was not properly set
        # during the initialisation (for instance, you entered Ommega_cdm
        # instead of Omega_cdm).  Then, the code exits, to prevent running with
        # imaginary parameters. This behaviour is also used in case you want to
        # kill the process.
        # - CosmoComputationError is returned if Class fails to compute the
        # output given the parameter values. This will be considered as a valid
        # point, but with minimum likelihood, so will be rejected, resulting in
        # the choice of a new point.
        if 'SZ' in data.cosmo_arguments['output']:
            try:
                if 'SZ_counts':
                    cosmo.compute(["szcount"])
                else:
                    cosmo.compute(["szpowerspectrum"])
            except CosmoComputationError as failure_message:
                sys.stderr.write(str(failure_message)+'\n')
                sys.stderr.flush()
                return data.boundary_loglike
            except CosmoSevereError as critical_message:
                raise io_mp.CosmologicalModuleError(
                    "Something went wrong when calling CLASS" +
                    str(critical_message))
            except KeyboardInterrupt:
                raise io_mp.CosmologicalModuleError(
                    "You interrupted execution")
        else:
            try:
                cosmo.compute(["lensing"])
            except CosmoComputationError as failure_message:
                # could be useful to uncomment for debugging:
                #np.set_printoptions(precision=30, linewidth=150)
                #print 'cosmo params'
                #print data.cosmo_arguments
                #print data.cosmo_arguments['tau_reio']
                sys.stderr.write(str(failure_message)+'\n')
                sys.stderr.flush()
                return data.boundary_loglike
            except CosmoSevereError as critical_message:
                raise io_mp.CosmologicalModuleError(
                    "Something went wrong when calling CLASS" +
                    str(critical_message))
            except KeyboardInterrupt:
                raise io_mp.CosmologicalModuleError(
                    "You interrupted execution")

    # For each desired likelihood, compute its value against the theoretical
    # model
    loglike = 0
    # This flag holds the information whether a fiducial model was written. In
    # this case, the log likelihood returned will be '1j', meaning the
    # imaginary number i.
    flag_wrote_fiducial = 0

    for likelihood in data.lkl.itervalues():
        if likelihood.need_update is True:
            value = likelihood.loglkl(cosmo, data)
            # Storing the result
            likelihood.backup_value = value
        # Otherwise, take the existing value
        else:
            value = likelihood.backup_value
        if data.command_line.display_each_chi2:
            print "-> for ",likelihood.name,":  loglkl=",value,",  chi2eff=",-2.*value
        loglike += value
        # In case the fiducial file was written, store this information
        if value == 1j:
            flag_wrote_fiducial += 1
    if data.command_line.display_each_chi2:
            print "-> Total:  loglkl=",loglike,",  chi2eff=",-2.*loglike

    # Compute the derived parameters if relevant
    if data.get_mcmc_parameters(['derived']) != []:
        try:
            derived = cosmo.get_current_derived_parameters(
                data.get_mcmc_parameters(['derived']))
            for name, value in derived.iteritems():
                data.mcmc_parameters[name]['current'] = value
        except AttributeError:
            # This happens if the classy wrapper is still using the old
            # convention, expecting data as the input parameter
            cosmo.get_current_derived_parameters(data)
        except CosmoSevereError:
            raise io_mp.CosmologicalModuleError(
                "Could not write the current derived parameters")
    for elem in data.get_mcmc_parameters(['derived']):
        data.mcmc_parameters[elem]['current'] /= \
            data.mcmc_parameters[elem]['scale']

    # If fiducial files were created, inform the user, and exit
    if flag_wrote_fiducial > 0:
        if flag_wrote_fiducial == len(data.lkl):
            raise io_mp.FiducialModelWritten(
                "This is not an error but a normal abort, because " +
                "fiducial file(s) was(were) created. " +
                "You may now start a new run. ")
        else:
            raise io_mp.FiducialModelWritten(
                "This is not an error but a normal abort, because " +
                "fiducial file(s) was(were) created. " +
                "However, be careful !!! Some previously non-existing " +
                "fiducial files were created, but potentially not all of them. " +
                "Some older fiducial files will keep being used. If you have doubts, " +
                "you are advised to check manually in the headers of the " +
                "corresponding files that all fiducial parameters are consistent "+
                "with each other. If everything looks fine, "
                "you may now start a new run.")

    return loglike/data.command_line.temperature


def compute_fisher(data, cosmo, center, step_size, step_matrix):
    # Adapted by T. Brinckmann
    parameter_names = data.get_mcmc_parameters(['varying'])
    fisher_matrix = np.zeros(
        (len(parameter_names), len(parameter_names)), 'float64')
    # Initialise the gradient field
    gradient = np.zeros(len(parameter_names), 'float64')

    # Re-center all parameters
    for elem in center:
        data.mcmc_parameters[elem]['current'] = center[elem]

    # Compute loglike at the point supposed to be a good estimate of the best-fit
    data.update_cosmo_arguments()
    loglike_min = compute_lkl(cosmo, data)

    # Loop through diagonal elements first, followed by off-diagonal elements
    for elem in ['diag','off-diag']:

        for k, elem_k in enumerate(parameter_names):
            kdiff = step_size[k]

            # loop over step direction
            # DEBUG: does step_index split cause a problem with Cholesky?
            # DEBUG: It causes a problem for all modes, it just more apparent with
            # DEBUG: the Cholesky method because of the large changes in stepsize
            # DEBUG: during subsequent iteration. The problem is the diagonal element
            # DEBUG: is computed incorrectly when the stepsize is iterated on.
            # DEBUG: The solution is to not use the step index until all diagonal
            # DEBUG: elements have been computed. This may be non-trivial to code,
            # DEBUG: but allowing off-diagonal elements to use step_index will
            # DEBUG: keep all of the speed-up and would still be correct.
            # DEBUG: FIXED (but what about with proper treatment of steps?)
            for step_index in [0,1]:
                if elem == 'diag' and step_index == 1:
                    continue
                # loop over second parameter
                for h, elem_h in enumerate(parameter_names):
                    hdiff = step_size[h]

                    # Since the matrix is symmetric, we only compute the
                    # elements of one half of it plus the diagonal.
                    if k > h:
                        continue
                    if k == h and elem == 'diag':
                        #print ''
                        print '---> Computing fisher element (%d,%d)' % (k,h)
                        temp1, temp2, diff_1 = compute_fisher_element(
                            data, cosmo, center, step_matrix, loglike_min, step_index,
                            (elem_k, kdiff))
                        fisher_matrix[k][k] += temp1
                        gradient[k] += temp2
                        step_size[k] = diff_1
                    elif k < h and elem == 'off-diag':
                        #print ''
                        print '---> Computing fisher element (%d,%d), part %d/2' % (k,h,step_index+1)
                        fisher_matrix[k][h] += compute_fisher_element(
                            data, cosmo, center, step_matrix, loglike_min, step_index,
                            (elem_k, kdiff),
                            (elem_h, hdiff))
                        fisher_matrix[h][k] = fisher_matrix[k][h]

    return fisher_matrix, gradient

def compute_fisher_element(data, cosmo, center, step_matrix, loglike_min, step_index_1, one, two=None):
    # Unwrap
    name_1, diff_1 = one
    if two:
        name_2, diff_2 = two

        if step_index_1 == 1:
            # Calculate step (++)
            step_index_2 = 1
            loglike_1, rotated_step_xx = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)

            # Calculate step (+-)
            step_index_2 = 0
            loglike_2, rotated_step_xy = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)
        else:
            loglike_1 = 0
            loglike_2 = 0

        if step_index_1 == 0:
            # Calculate step (-+). Assuming symmetry rotated_step_yx = rotated_step_xy
            step_index_2 = 1
            loglike_3, rotated_step_xy = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)

            # Calculate step (--). Assuming symmetry rotated_step_yy = rotated_step_xx
            step_index_2 = 0
            loglike_4, rotated_step_xx = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)
        else:
            loglike_3 = 0
            loglike_4 = 0

        # If the left and right step sizes are equal these terms will cancel
        if abs(diff_2[0]) == abs(diff_2[1]) or diff_2[2] or data.use_symmetric_step:
            loglike_5 = 0.
            loglike_6 = 0.
        else:
            if step_index_1 == 1:
                # Calculate step (+/)
                step_index_2 = None
                loglike_5, rotated_step_x = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)
            else:
                loglike_5 = 0

            if step_index_1 == 0:
                # Calculate step (-/)
                step_index_2 = None
                loglike_6, rotated_step_x = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)
            else:
                loglike_6 = 0

        # If the left and right step sizes are equal these terms will cancel
        if abs(diff_1[0]) == abs(diff_1[1]) or diff_1[2] or data.use_symmetric_step:
            loglike_7 = 0.
            loglike_8 = 0.
        else:
            step_index_1 = None
            # Calculate step (/+)
            step_index_2 = 1
            loglike_7, rotated_step_y1 = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)

            # Calculate step (/-)
            step_index_2 = 0
            loglike_8, rotated_step_y2 = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)

        # Count the bestfit term at most once
        if step_index_1:
            loglike_min = 0

        # In the following we want only the step magnitude, not sign
        diff_1_backup = diff_1[:]
        diff_2_backup = diff_2[:]
        diff_1 = abs(diff_1)
        diff_2 = abs(diff_2)
        # Remember that in some cases only used diff_1[2] or diff_1[0]
        if diff_1[2]:
            diff_1[0] = diff_1[2]
            diff_1[1] = diff_1[2]
        elif data.use_symmetric_step:
            diff_1[1] = diff_1[0]
        # And only used diff_2[2] or diff_2[0]
        if diff_2[2]:
            diff_2[0] = diff_2[2]
            diff_2[1] = diff_2[2]
        elif data.use_symmetric_step:
            diff_2[1] = diff_2[0]
        
        # For Fisher mode 1 and 2 the stepsizes are more complicated.
        # We need to compute the magnitude of the array (A * x), where
        # A is the rotation matrix (i.e. the Cholesky decomposition or the
        # eigenvectors of the covariance matrix) and x is the step array
        # (remember the full step would be center + rotation_matrix*step_array).
        # After computing the Fisher matrix we then need to rotate back to
        # parameter space, this is done in the get_fisher_matrix function.
        parameter_names = data.get_mcmc_parameters(['varying'])
        index_1 = parameter_names.index(name_1)
        index_2 = parameter_names.index(name_2)
        #if data.fisher_mode == 1:
        #    diff_1 *= step_matrix[index_1,index_1]**-1.
        #    diff_1 *= step_matrix[index_2,index_2]**-1.
        if data.fisher_mode == 2 or data.fisher_mode == 1:
            #diff_1 *= step_matrix[index_1,index_1]
            #diff_2 *= step_matrix[index_2,index_2]
            # Re-compute the rotated step. Unfortunately we can't just use the output
            # from compute_fisher_step because we need the two steps separated.
            step_array_1 = np.zeros(len(parameter_names))
            step_array_1[index_1] = diff_1[0]
            if data.fisher_mode == 2:
                rotated_step_1 = np.dot(step_matrix[:,index_1],step_array_1)**0.5
            else:
                rotated_step_1 = np.dot(step_matrix[:,index_1],step_array_1*step_matrix[index_1,index_1]**-1.)**0.5
            step_array_2 = np.zeros(len(parameter_names))
            step_array_2[index_2] = abs(diff_2[0])
            if data.fisher_mode == 2:
                rotated_step_2 = np.dot(step_matrix[:,index_2],step_array_2)**0.5
            else:
                rotated_step_2 = np.dot(step_matrix[:,index_2],step_array_2*step_matrix[index_2,index_2]**-1.)**0.5
            # Instead of a simple stepsize (e.g. diagonal element of covariance matrix)
            # we need to instead consider that we rotated our step in parameter space,
            # and instead use the magnitude of the rotated step vector.
            diff_1[0] = np.dot(abs(rotated_step_1),abs(rotated_step_1))**0.5
            diff_2[0] = np.dot(abs(rotated_step_2),abs(rotated_step_2))**0.5

            # The next part is if steps aren't symmetric, but in this part we had assumed
            # symmetric steps for the derivation of the derivative, so this likely
            # wouldn't work without additional modifications.
            if not data.use_symmetric_step:
                warnings.warn('Current implementation assumes symmetric step, this probably will not work')
                step_array_1 = np.zeros(len(parameter_names))
                step_array_1[index_1] = diff_1[1]
                if data.fisher_mode == 2:
                    rotated_step_1 = np.dot(step_matrix[:,index_1],step_array_1)**0.5
                else:
                    rotated_step_1 = np.dot(step_matrix[:,index_1],step_array_1*step_matrix[index_1,index_1]**-1.)**0.5
                step_array_2 = np.zeros(len(parameter_names))
                step_array_2[index_2] = abs(diff_2[1])
                if data.fisher_mode == 1:
                    rotated_step_2 = np.dot(step_matrix[:,index_2],step_array_2)**0.5
                else:
                    rotated_step_2 = np.dot(step_matrix[:,index_2],step_array_2*step_matrix[index_2,index_2]**-1.)**0.5
                diff_1[1] = np.dot(abs(rotated_step_1),abs(rotated_step_1))**0.5
                diff_2[1] = np.dot(abs(rotated_step_2),abs(rotated_step_2))**0.5
            else:
                diff_1[1] = diff_1[0]
                diff_2[1] = diff_2[0]
            #diff_1 *= np.dot(step_matrix[index_1:,index_1],step_matrix[index_1:,index_2])**0.5
            #diff_2 *= np.dot(step_matrix[index_2:,index_2],step_matrix[index_2:,index_2])**0.5

            #diff_1 *= np.dot(step_matrix[index_1:,index_1],step_matrix[index_2:,index_2])**-0.5
            #diff_2 *= np.dot(step_matrix[index_2:,index_2],step_matrix[index_1:,index_1])**-0.5
        #fisher_off_diagonal = -(
        #    loglike_1-loglike_2-loglike_3+loglike_4)/(4.*diff_1*diff_2)

        # Fisher mode 2 only considers the covariance of parameters in two dimensions at a time.
        # DEBUG: Maybe for the stepsize we need to compute the magnitude of the rotated step?
        # DEBUG: Would we need to rotate back afterwards?
        if data.fisher_mode == 3:
            # Fisher mode 3 assumes symmetry, so loglike_5, 6, 7 and 8 are zero and the step size for
            # loglike_1 (++) and loglike_4 (--), as well as loglike_2 (+-) and loglike_3 (-+), are the same.
            # The computation can therefore be split into two parts, one for loglike_1 and loglike_4 and one
            # for loglike_2 and loglike_3, with different step sizes for each pair corresponding to the rotated steps.
            # loglike_1 (step_index_2 = 1) used step "xx", loglike_2 (step_index_2 = 0) used step "xy"
            # loglike_3 (step_index_2 = 1) used step "xy" and loglike_4 (step_index_2 = 0) used step "xx"
            fisher_off_diagonal = -((1./(4*abs(rotated_step_xx[index_1])*abs(rotated_step_xx[index_2]))) * (loglike_1 + loglike_4)
                                    + (1./(4*abs(rotated_step_xy[index_1])*abs(rotated_step_xy[index_2]))) * (- loglike_2 - loglike_3))
        # In the case of symmetric steps reduces to -(loglike_1-loglike_2-loglike_3+loglike_4)/(4.*diff_1*diff_2)
        else:
            # DEBUG: for Cholesky step, diff_1 and diff_2 should be magnitudes of step array.
            # DEBUG: problem: rotated_step is for each combination of two diff arrays, and we need the four diff arrays separately!
            # DEBUG: However, instead of arrays we can make an approximation:
            # DEBUG: We could do something similar to fisher mode 3 for the step sizes of the general case. This
            # DEBUG: would be an approximation, since it would assume only steps in two parameters, when several
            # DEBUG: parameters may have changed. However, it is probably better than the current implementation.
            # DEBUG: Note that it only works if we assume symmetry! Otherwise either the step_size=[0,1] scheme
            # DEBUG: needs to be abandoned (slowing down the code a lot!) or rotated_step_xx/xy/yx/yy and loglike
            # DEBUG: values need to be stored and the fisher matrix assembled at the end. Also, loglike_5,6,7,8
            # DEBUG: may not be zero in that case.
            fisher_off_diagonal = -((1./(diff_2[0]**2./diff_2[1]+diff_2[0]))* # sym. \Delta p_j: reduces to 1/(2 \Delta p_j)
                                    (1./(diff_1[0]**2./diff_1[1]+diff_1[0]))* # sym. \Delta p_i: reduces to 1/(2 \Delta p_i)
                                    ((diff_2[0]/diff_2[1])**2. * ((diff_1[0]/diff_1[1])**2.*loglike_1 - loglike_3) # sym. \Delta p_i and \Delta p_j: reduces to loglike_1 - loglike_3
                                     -(diff_1[0]/diff_1[1])**2.*loglike_2 + loglike_4 # sym \Delta p_i: reduces to loglike_4 - loglike_2
                                     +((diff_2[0]/diff_2[1])**2.-1.) * (loglike_6 - (diff_1[0]/diff_1[1])**2.*loglike_5 + ((diff_1[0]/diff_1[1])**2.-1.)*loglike_min) # cancels if sym. \Delta p_j
                                     +((diff_1[0]/diff_1[1])**2.-1.) * (loglike_8 - (diff_2[0]/diff_2[1])**2.*loglike_7))) # cancels if sym. \Delta p_i

        # Restore step sign
        diff_1 = diff_1_backup
        diff_2 = diff_2_backup

        return fisher_off_diagonal
    # It is otherwise a diagonal component
    else:
        # Calculate left step
        step_index_1 = 0
        step_index_2 = None
        loglike_left, diff_1, rotated_step_left = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)
        # Update 'one' with the new step size to pass to the next step
        one = (one[0],diff_1)

        # Calculate right step
        step_index_1 = 1
        loglike_right, diff_1, rotated_step_right = compute_fisher_step(data,cosmo,center,step_matrix,loglike_min,one,two,step_index_1,step_index_2)

        # In the following we want only the step magnitude, not sign
        diff_1_backup = diff_1[:]
        diff_1 = abs(diff_1)
        # Remember that in some cases only used diff_1[2] or diff_1[0]
        if diff_1[2]:
            diff_1[0] = diff_1[2]
            diff_1[1] = diff_1[2]
        elif data.use_symmetric_step:
            diff_1[1] = diff_1[0]

        # For fisher mode 3 we need to use the rotated step size instead of the input step size
        parameter_names = data.get_mcmc_parameters(['varying'])
        index = parameter_names.index(name_1)
        #if data.fisher_mode == 1:
        #    diff_1 *= step_matrix[index,index]**-1.
        if data.fisher_mode == 3:
            # Due to symmetry assumed for fisher mode 3 the following two quantities should be equal
            diff_1[0] = abs(rotated_step_left[index])
            diff_1[1] = abs(rotated_step_right[index])
            #print 'steps equal?', diff_1[0]==diff_1[1]
            #print 'rotated array, rotated step =', rotated_step_left,rotated_step_left[index]
        elif data.fisher_mode > 0:
            #diff_1 *= step_matrix[index,index]
            #print step_matrix[index:,index]
            #print 'diag =',step_matrix[index,index]
            #print 'modulus =',np.dot(step_matrix[index:,index],step_matrix[index:,index])**0.5
            #print diff_1
            #print 'old diff_1 ='
            #for i in range(len(diff_1)):
            #    print diff_1[i]*step_matrix[index,index]
            #diff_1 *= np.dot(step_matrix[index:,index],step_matrix[index:,index])**0.5
            #print 'alternate diff_1[0] =',diff_1[0]*np.dot(step_matrix[index:,index],step_matrix[index:,index])**0.5
            #diff_1 /=np.dot(step_matrix[index:,index],step_matrix[index:,index])**0.5
            diff_1[0] = (np.dot(abs(rotated_step_left),abs(rotated_step_left)))**0.5
            diff_1[1] = (np.dot(abs(rotated_step_right),abs(rotated_step_right)))**0.5
            #print 'new diff_1 ='
            #for i in range(len(diff_1)):
            #    print diff_1[i]
            ##diff_1_backup2 = diff_1[:]
            ##diff_1 = [[],[]]
            # DEBUG: For Cholesky step, diff_1 should be an array, not scalars.
            # DEBUG: However, instead of arrays we can make an approximation and use the same scheme
            # DEBUG: as for fisher mode 3. Although more parameters are changed, it might be a good 
            # DEBUG: approximation to take the rotated step left/right here as well.
            #diff_1[0] = abs(rotated_step_left[index])
            #diff_1[1] = abs(rotated_step_right[index])

        #fisher_diagonal = -(loglike_right-2.*loglike_min+loglike_left)/(diff_1**2)
        # In case of symmetric steps reduces to -(loglike_right-2.*loglike_min+loglike_left)/(diff_1**2)
        fisher_diagonal = -2.*((diff_1[0]/diff_1[1])*loglike_right-(diff_1[0]/diff_1[1]+1.)
                                *loglike_min+loglike_left)/(diff_1[0]*diff_1[1]+diff_1[0]**2.)
        #print 'fisher_diag =',fisher_diagonal,' - loglike left, min, right =', loglike_left, loglike_min, loglike_right

        ##fisher_diagonal = -2.*(np.dot(abs(diff_1[0]),abs(1./diff_1[1]))*loglike_right-(np.dot(abs(diff_1[0]),abs(1./diff_1[1]))+1.)
        ##                        *loglike_min+loglike_left)/(np.dot(abs(diff_1[0]),abs(diff_1[1]))+np.dot(diff_1[0],diff_1[0]))

        ##diff_1 = diff_1_backup2

        #gradient = -(loglike_right-loglike_left)/(2.*diff_1)
        # In case of symmetric steps reduces to -(loglike_right-loglike_left)/(2.*diff_1)
        gradient = -((diff_1[0]/diff_1[1])**2.*loglike_right-loglike_left-((diff_1[0]/diff_1[1])**2.-1.)
                     *loglike_min)/((diff_1[0]**2./diff_1[1])+diff_1[0])

        # Restore step sign
        diff_1 = diff_1_backup

        return fisher_diagonal, gradient, diff_1


def compute_fisher_step(data, cosmo, center, step_matrix, loglike_min, one, two, step_index_1, step_index_2):
    name_1, diff_1 = one
    if two:
        name_2, diff_2 = two

    deltaloglkl_req = data.fisher_delta
    deltaloglkl_tol = data.fisher_tol

    # Create an array of the center value
    parameter_names = data.get_mcmc_parameters(['varying'])
    center_array = np.zeros(len(center), 'float64')
    for elem in parameter_names:
        index = parameter_names.index(elem)
        center_array[index] = center[elem]

    if two and data.fisher_mode == 3:
        index1 = parameter_names.index(name_1)
        index2 = parameter_names.index(name_2)
        step_sub_matrix = np.zeros([2,2])
        step_sub_matrix[0,0] = step_matrix[index1,index1]
        step_sub_matrix[1,1] = step_matrix[index2,index2]
        step_sub_matrix[0,1] = step_matrix[index1,index2]
        step_sub_matrix[1,0] = step_matrix[index2,index1]
        step_matrix = np.zeros(np.shape(step_matrix))
        step_matrix[index1,index1] = step_sub_matrix[0,0]
        step_matrix[index2,index2] = step_sub_matrix[1,1]
        step_matrix[index1,index2] = step_sub_matrix[0,1]
        step_matrix[index2,index1] = step_sub_matrix[1,0]
    elif data.fisher_mode == 3:
        index1 = parameter_names.index(name_1)
        backup_step = step_matrix[index1,index1]
        step_matrix = np.zeros(np.shape(step_matrix))
        step_matrix[index1,index1] = backup_step

    backup_step = [0.]
    repeat = 1
    while repeat:
        # Create array with new steps in the basis of the parameters
        step_array = np.zeros(len(parameter_names), 'float64')
        if not step_index_1 == None:
            index = parameter_names.index(name_1)
            norm = 1.
            # If we are changing two parameters we need to add a normalization 1/sqrt(2)
            if two and not step_index_2 == None:
                # DEBUG: why normalize by number of parameters? (it was done in the mcmc)
                # DEBUG: norm = 2 doesn't seem to work
                #norm = 2.
                norm = 1.

            # Check if the parameter step had exceeded a boundary.
            # Assume symmetric likelihood and use opposite step if so.
            # I.e. both steps (+/-) will be the same and will return the same -loglkl.
            if diff_1[2]:
                step_array[index] = diff_1[2]/norm**0.5
            # If symmetric step is required use diff_1[0] to determine size of step.
            # Only step_index_1=0 goes through the iteration cycle in this case.
            # If diff_1[2] is defined, will instead use that value.
            elif data.use_symmetric_step:
                step_array[index] = np.sign(diff_1[step_index_1])*abs(diff_1[0])/norm**0.5
            else:
                step_array[index] = diff_1[step_index_1]/norm**0.5

            # If we don't want to use the Cholesky to determine stepsize, instead
            # use input stepsize normalized by the diagonal element of the Cholesky.
            #if not data.use_cholesky_step and data.fisher_mode > 1:
            if not data.use_cholesky_step:
                step_array[index] *= step_matrix[index,index]**-1

        if two and not step_index_2 == None:
            index = parameter_names.index(name_2)
            # We are changing two parameters so we need to add a normalization 1/sqrt(2)
            # DEBUG: why normalize by number of parameters? (it was done in the mcmc)
            # DEBUG: norm = 2 doesn't seem to work
            #norm = 2.
            norm = 1.

            # Check if the parameter step had exceeded a boundary.
            # Assume symmetric likelihood and use opposite step if so.
            # I.e. both steps (+/-) will be the same and will return the same -loglkl.
            if diff_2[2]:
                step_array[index] = diff_2[2]/norm**0.5
            # If symmetric step is required use diff_2[0] to determine size of step.
            # Only step_index_2=0 goes through the iteration cycle in this case.
            # If diff_2[2] is defined, will instead use that value.
            elif data.use_symmetric_step:
                step_array[index] = np.sign(diff_2[step_index_2])*abs(diff_2[0])/norm**0.5
            else:
                step_array[index] = diff_2[step_index_2]/norm**0.5

            # If we don't want to use the Cholesky to determine stepsize, instead
            # use input stepsize normalized by the diagonal element of the Cholesky.
            #if not data.use_cholesky_step and data.fisher_mode > 1:
            # DEBUG: Trying out normalizing the eigenvectors by the diagonal element,
            # DEBUG: but this shouldn't really matter, as we're just changing the
            # DEBUG: starting point of the step iteration.
            if not data.use_cholesky_step:
                step_array[index] *= step_matrix[index,index]**-1.

        # Rotate the step vector to the basis of the covariance matrix
        # DEBUG: I forgot, why square root of the rotated array?
        # DEBUG: To account for the dimension of the step matrix elements
        # DEBUG: The square root of the step_array part is irrelevant for
        # DEBUG: cholesky step, it's "just a number"
        if data.use_cholesky_step and data.fisher_mode > 1:
            rotated_array = np.dot(step_matrix, step_array)
            rotated_array = np.sign(rotated_array)*abs(rotated_array)**0.5
        else:
            rotated_array = np.dot(step_matrix, step_array)

        # Construct step vector for comparison with previous step
        step_array = center_array + rotated_array

        # Check for slow/fast parameters, comparing to the last step in the Fisher matrix calculation.
        # This means calling CLASS can be skipped if only nuisance parameters changed.
        data.check_for_slow_step(step_array)

        # In order to take the correct new step we need to re-center
        # the parameters and add the rotated step vector.
        for elem in center:
            index = parameter_names.index(elem)
            data.mcmc_parameters[elem]['current'] = center[elem] + rotated_array[index]

        #print 'Need cosmo update:', data.need_cosmo_update
        # Update current parameters to the new parameters, only taking steps as requested
        data.update_cosmo_arguments()

        # Compute loglike value for the new parameters
        loglike = compute_lkl(cosmo, data)

        # Iterative stepsize. If -Delta ln(L) > 1, change step size and repeat steps above
        # if data.use_symmetric_step=True only runs for step_index_1=0
        # ISSUE: what about Cholesky boundaries?
        if not two and not (data.use_symmetric_step and step_index_1 == 1):
            # Save previous step
            # For symmetric step
            if diff_1[2]:
                backup_step.append(diff_1[2])
            # For normal step
            else:
                backup_step.append(diff_1[step_index_1])

            # Calculate Delta ln(L)
            Deltaloglike = loglike - loglike_min
            #print ">>>> For %s[%d],Delta ln(L)=%e using min(ln(L))=%e"%(name_1,int(np.sign(diff_1[step_index_1])),loglike-loglike_min,loglike_min)

            # If -Delta ln(L) is larger than desired reduce stepsize
            if Deltaloglike < -(deltaloglkl_req + deltaloglkl_tol):
                if diff_1[2]:
                    diff_1[2] -= np.sign(diff_1[2]) * 0.5 * abs(backup_step[abs(repeat)-1] - diff_1[2])
                else:
                    diff_1[step_index_1] -= np.sign(diff_1[step_index_1]) * 0.5 * abs(backup_step[abs(repeat)-1] - diff_1[step_index_1])
                #print 'Updated stepsize. Before, after =',backup_step[abs(repeat)],diff_1[step_index_1]
                repeat = len(backup_step)

            # If -Delta ln(L) is smaller than desired increase stepsize
            # ISSUE: what about boundaries when increasing stepsize?
            elif Deltaloglike > -(deltaloglkl_req - deltaloglkl_tol):
                if repeat > 1:
                    if diff_1[2]:
                        diff_1[2] -= np.sign(diff_1[2]) * 0.5 * abs(backup_step[abs(repeat)-1] - diff_1[2])
                    else:
                        diff_1[step_index_1] += np.sign(diff_1[step_index_1]) * 0.5 * abs(backup_step[abs(repeat)-1] - diff_1[step_index_1])
                    #print 'Updated stepsize. Before, after =',backup_step[abs(repeat)],diff_1[step_index_1]
                    repeat = len(backup_step)
                else:
                    if diff_1[2]:
                        diff_1[2] *= 2.
                    else:
                        diff_1[step_index_1] *= 2.
                    #print 'Updated stepsize. Before, after =',backup_step[abs(repeat)],diff_1[step_index_1]
                    repeat = -len(backup_step)
            else:
                return loglike, diff_1, rotated_array
        elif not two:
            return loglike, diff_1, rotated_array
        else:
            return loglike, rotated_array


def adjust_fisher_bounds(data, center, step_size):
    # For the Fisher approach we may need to adjust the step size if the step
    # exceed the bounds on the parameter given in the param file. We Loop through
    # all parameters, adjusting the step size of any parameter where that step
    # exceeded the bounds.
    for index, elem in enumerate(data.get_mcmc_parameters(['varying'])):
        param = data.mcmc_parameters[elem]['initial']

        if param[1] != None:
            if param[1] > center[elem]:
                raise io_mp.ConfigurationError("Error in parameter ranges: left edge %e bigger than central value %e.\n"
                                               %(param[1],center[elem]))
            # When encountering a boundary, set stepsize to boundary limit
            if param[1] > center[elem] + step_size[index,0]:
                step_size[index,0] = -(center[elem] - param[1])
                # Instead of asymmetric steps, assumme symmetric likelihood and use positive step
                if not param[2] < center[elem] + step_size[index,1]:
                    step_size[index,2] = step_size[index,1]
                    print 'Negative step exceeded boundary for',elem,'- using symmetry assumption with stepsize =',step_size[index,2]

        if param[2] != None:
            if param[2] < center[elem]:
                raise io_mp.ConfigurationError("Error in parameter ranges: right edge %e smaller than central value %e.\n"
                                               %(param[2],center[elem]))
            # When encountering a boundary, set stepsize to boundary limit
            if param[2] < center[elem] + step_size[index,1]:
                step_size[index,1] = param[2] - center[elem]
                # Instead of asymmetric steps, assumme symmetric likelihood and use negative step
                if not param[1] > center[elem] + step_size[index,0]:
                    step_size[index,2] = step_size[index,0]
                    print 'Positive step exceeded boundary for',elem,'- using symmetry assumption with stepsize =',step_size[index,2]

        # If we want to use the Cholesky to determine stepsizes, normalize step_size to 1
        # ISSUE: what about when the Cholesky step (rather than parameter basis step) exceeds the boundary?
        # TODO: proper test for Cholesky step boundary.
        if data.use_cholesky_step and data.fisher_mode > 1:
            step_size[index,0] = -1.
            step_size[index,1] = 1.
            step_size[index,2] = np.sign(step_size[index,2])

    return step_size


def vectorize_dictionary(data, center, one, two, step_index_1, step_index_2):
    name_1, diff_1 = one
    if two:
        name_2, diff_2 = two

    # In order to compare the last step to the current step via
    # data.check_for_slow_step we need the parameters ordered
    # correctly as an array.
    parameter_names = data.get_mcmc_parameters(['varying'])
    step_vector = np.zeros(len(center))
    for elem in parameter_names:
        index = parameter_names.index(elem)
        step_vector[index] = center[elem]
        if not step_index_1 == None:
            if elem == name_1:
                step_vector[index] += diff_1[step_index_1]
        if not step_index_2 == None:
            if elem == name_2:
                step_vector[index] += diff_2[step_index_2]

    return step_vector
