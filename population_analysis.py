############################## Change Parameter Here ##############################
# parameters for the ensemble sampler
nwalkers = 100 # The number of walkers in the ensemble
ndim = 4 # The number of dimensions in the parameter space
param_initial_guess = [1.0, 1.5, 5.0, 90.0]
backend_file = "output.h5" # output file name

############################## Fetch Data File ##############################
import json # to read JSON file
import requests # to download data file via URL
import os.path


# opening JSON file
event_list_file = open('event_list.json', "r")

# return "files" element in JSON object as a dictionary
event_list = (json.load(event_list_file))['files']

for event in event_list[:10]: # Here we only loop first few files for testing, remove "[:10]" in actual use
    if event['type'] == 'h5': # Check if the event links to a H5 file
        if (event['key'][-10:]) != 'nocosmo.h5': # We only want cosmological reweighted data
            url = event['links']['self'] # get the url
            filename = event['key'] # get the file name
            
            if os.path.isfile('data/' + filename) == False: # if the file does not exist
                print('Downloading ' + filename)
                r = requests.get(url, allow_redirects=True)
                open('data/' + filename, 'wb').write(r.content) # download the data file into the data folder
            else: # if the already exist
                print(filename + ' exists')


############################## Obtain Posterior Samples ##############################
import numpy as np
import pandas as pd


def get_posterior_samples(file_name):
    return pd.read_hdf(file_name, key="C01:Mixed/posterior_samples")

#a = get_posterior_samples('data/IGWN-GWTC3p0-v1-GW191126_115259_PEDataRelease_mixed_cosmo.h5', 'mass_1_source')
#print(a)

posterior_samples = [] # create an empty list

for file in os.listdir('data'): # loop through files in the data folder
    posterior_samples.append(get_posterior_samples('data/' + file)) # append the address of dataframe to the list


############################## Obtain Prior Function ##############################
def get_prior(parameter):
    return 1

############################## Utility ##############################
# Smoothing Function
def smoothing_function(m, m_min, delta):
    if m < m_min:
        return 0.0
    elif m_min <= m and m < (m_min + delta):
        return 1.0/(np.exp((delta / (m - m_min)) + (delta/ (m-m_min-delta))) + 1.0)
    elif m >= (m_min + delta):
        return 1.0

    
############################## Hierarchical Bayesian Analysis ##############################
# Truncated power law model: data[0] = m_1, data[1] = q, params[0] = alpha, params[1] = beta, params[2]=m_min, params[3]=m_max
def power_law(data, params):
    m_1, q = data[0], data[1]
    alpha, beta, m_min, m_max = params[0], params[1], params[2], params[3]
    epsilon = 0.001 # a very small number for limit computation
    if m_1 > m_min and m_1 < m_max: # if m_1 is within the range bounded by m_min and m_max
        normalization_constant = 1.0
        if alpha < (1.0 + epsilon) and alpha > (1.0 - epsilon): # For special case where alpha is in the proximity of 1
            normalization_constant *= np.log(m_max / m_min)
        else:
            normalization_constant *= (m_max ** (1 - alpha) - m_min ** (1 - alpha)) / (1 - alpha)
        
        if beta > (-1.0 -epsilon) and beta < (-1.0 + epsilon):
            return 0.0 # The normalization constant will be negative infinity in this case
        else:
            normalization_constant *= (1 / (beta + 1))
        
        return ((m_1 ** (-alpha)) / normalization_constant)
    else:
        return 0.0

def power_law_plus_peak(data, params):
    m_1, q = data[0], data[1]
    alpha, beta, m_min, m_max, lambda_p, mu, sigma, delta = params[0], params[1]. params[2], params[3], params[4], params[5], params[6], params[7]
    epsilon = 0.001 # a very small number for limit computation
    
    
    
    
# Broken power law model, params = [0: alpha, 1: beta, 2: m_min, 3: m_max, 4: lambda, 5: mu, 6: sigma, 7: delta]
def broken_power_law(m_1, params):
    # TODO
    
    
    return 1

# log probability distribution on population parameters
# Model: "truncated_power_law"
# Truncated Power Law: params[0] = alpha, params[1] = beta, params[2]=m_min, params[3]=m_max
def log_population_distribution(params, model): 
    # check on parameters
    if (model=="truncated_power_law") * (params[1] < 0) * (params[2] < 0): # mass cannot be smaller or equal to zero
        return -np.inf
    elif (model=='broken_power_law'):
        # TODO:
        
        return -np.inf
    # if parameters are ok, do the computation
    else:
        log_population_distribution = 0.0 # initialize the value to zero
        for event in posterior_samples:
            sum = 0.0
            for i in range(event.shape[0]):
                if (model=="truncated_power_law"):
                    data = np.array([event['mass_1_source'][i], event['mass_ratio'][i]])
                    sum += power_law(data, params)
                elif (model == "broken_power_law"):
                    # TODO:
                    sum += 0
                    
            log_population_distribution += np.log(sum) - np.log(event.shape[0]) # sum divided by the number of samples                     
        if np.isfinite(log_population_distribution):
            return log_population_distribution
        else:
            return -np.inf



############################## MCMC to sample the posterior ##############################
import emcee
import corner
import matplotlib.pyplot as plt

np.random.seed(0)

initial_state = np.random.randn(nwalkers, ndim)

for walker in initial_state:
    for i in range(len(param_initial_guess)):
        walker[i] += param_initial_guess[i] # initial prediction
    

nsteps = 1000 # The number of steps to run

# Set up the backend
# Don't forget to clear it in case the file already exists
backend = emcee.backends.HDFBackend(backend_file)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=log_population_distribution, args=['truncated_power_law'], backend=backend)
# args is a list of extra arguments for log_prob_fn, log_prob_fn will be called with the sequence log_pprob_fn(p, *args, **kwargs)

sampler.run_mcmc(initial_state, nsteps)


############################## Output Graph ##############################
# plot graph
# Todo: need change to 4 dimension





            



