############################## Change Parameter Here ##############################
# parameters for the ensemble sampler
nwalkers = 100 # The number of walkers in the ensemble
ndim = 8 # The number of dimensions in the parameter space
nsteps = 1000 # The number of steps to run
param_initial_guess = [2.5, 1.2, 4.8, 4.59, 86.0, 0.1, 33.0, 5.7]
# param_initial_guess = [2.5, 6.0, 4.5, 80.0]
backend_file = "output.h5" # output file name
model = 'power_law_plus_peak'

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
        if (event['key'][-14:]) == 'mixed_cosmo.h5': # We only want cosmological reweighted data
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
import h5py


def get_posterior_samples(file_name):
    return h5py.File(file_name)["C01:Mixed/posterior_samples"]

#a = get_posterior_samples('data/IGWN-GWTC3p0-v1-GW191126_115259_PEDataRelease_mixed_cosmo.h5', 'mass_1_source')
#print(a)

posterior_samples = [] # create an empty list

for file in os.listdir('data'): # loop through files in the data folder
    posterior_samples.append(get_posterior_samples('data/' + file)) # append the address of dataframe to the list


############################## Obtain Prior Function ##############################
def get_prior(parameter):
    return 1

############################## Utility Function ##############################
# Smoothing Function
# Normally, m would come as an array of double, m_min and delta would come as double
def smoothing_function(m, m_min, delta):
    condition_list = [m < m_min, (m_min <= m) & (m < (m_min + delta)), m >= (m_min + delta)]
    choice_list = [0.0, 1.0/(np.exp((delta / (m - m_min)) + (delta/ (m-m_min-delta))) + 1.0), 1.0]
    return np.select(condition_list, choice_list)

def log_uniform_prior(min, max, x):
    if (x < min) | (x > max):
        return -np.infty
    else:
        return 0
   
# normalized Gaussian Distribution 
def normal_distribution(mean, sd, x):
    return (1/sd/np.sqrt(2*np.pi))*np.exp(-0.5*(((x-mean)/sd)**2))

# Heaviside step function
def step_function(x):
    return np.where(x >= 0, 1.0, 0.0)


    

############################## Prior Functions ##############################
# Log Prior of Truncated Power Law model
def power_law_prior(params):
    alpha, beta, m_min, m_max = params[0], params[1], params[2], params[3] # alpha, beta,... are double
    output = log_uniform_prior(-4., 12., alpha) + log_uniform_prior(-4., 12., beta) + log_uniform_prior(2.,10.,m_min) + log_uniform_prior(30.,100.,m_max)
    return output    
        

    

############################## Population Models ##############################
from scipy.integrate import quad
# Truncated power law model: data[0] = m_1, data[1] = q, params[0] = alpha, params[1] = beta, params[2]=m_min, params[3]=m_max
def power_law(data, params):
    m_1, q = data[0], data[1] # m_1 and q are arrays of values
    alpha, beta, m_min, m_max = params[0], params[1], params[2], params[3] # alpha, beta,... are double
    epsilon = 0.001 # a very small number for limit computation

    normalization_constant = 1.0
    if (alpha>(1.0-epsilon))&(alpha<(1.0+epsilon)):
        normalization_constant *= np.log(m_max/m_min)
    else:
        normalization_constant *= (m_max**(1.0-alpha)-m_min**(1.0-alpha))/(1.0-alpha)
    
    if (beta > (-1.0-epsilon)) & (beta<(-1.0+epsilon)):
        return 0.0 # The normalization constant will be negative infinity, this gives 0
    else:
        normalization_constant *= (1.0 / (beta + 1.0))
    
    return np.where((m_1 > m_min) & (m_1 < m_max),
                    (m_1 ** (-alpha)) * (q ** beta) / normalization_constant,
                    0.0)


def power_law_plus_peak(data, params):
    m_1, q = data[0], data[1]
    alpha, beta, delta, m_min, m_max, lambda_p, mu, sigma = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]

    epsilon = 0.001 # a very small number for limit computation
    
    if (alpha <= 1.0): 
        return 0 # if alpha is smaller of equal to 1.0, the area under the curve will be infinitely large
    
    
    if (beta >(-1.0-epsilon)) & (beta<(-1.0+epsilon)):
        normal_const = np.log(m_1/(m_min+delta))
    else:
        normal_const = (1-((m_min+delta)/m_1)**(beta+1))/ (1+beta)
    
    mass_prob = smoothing_function(m_1, m_min, delta)*((1-lambda_p)*(m_1**(-alpha))*step_function(m_max-m_1)*(alpha-1)/((m_max-m_1)**(1-alpha)) + lambda_p*normal_distribution(mu, sigma, m_1))
    
    q_prob = smoothing_function(m_1 * q, m_min, delta) * q**beta / normal_const
    return mass_prob * q_prob
    
    
    
    
# Broken power law model, params = [0: alpha, 1: beta, 2: m_min, 3: m_max, 4: lambda, 5: mu, 6: sigma, 7: delta]
def broken_power_law(m_1, params):
    # TODO
    
    
    return 1


############################## Log Likelihood Function ##############################
# log probability distribution on population parameters
# Model: "truncated_power_law"
# Truncated Power Law: params[0] = alpha, params[1] = beta, params[2]=m_min, params[3]=m_max
def log_population_distribution(params, model): 
    # check on population parameters
    if (model=="truncated_power_law"): # mass cannot be smaller or equal to zero
        population_prior = power_law_prior(params)
    elif (model=='power_law_plus_peak'):
        # TODO:
        population_prior = 0.0
    
    # if parameters are ok, do the computation
    log_population_distribution = population_prior # initialize the value to zero
    for event in posterior_samples:
        if (model == "truncated_power_law"):
            data = np.array([event['mass_1_source'], event['mass_ratio']])
            sum = np.sum(power_law(data, params))
        elif (model == "power_law_plus_peak"):
            data = np.array([event['mass_1_source'], event['mass_ratio']])
            sum = np.sum(power_law_plus_peak(data, params))
                    
        log_population_distribution += (np.log(sum) - np.log(event.shape[0])) # sum divided by the number of samples                     
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
    



# Set up the backend
# Don't forget to clear it in case the file already exists
backend = emcee.backends.HDFBackend(backend_file)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=log_population_distribution, args=[model], backend=backend)
# args is a list of extra arguments for log_prob_fn, log_prob_fn will be called with the sequence log_pprob_fn(p, *args, **kwargs)

sampler.run_mcmc(initial_state, nsteps, progress=True)


############################## Output Graph ##############################
# plot graph
# Todo: need change to 4 dimension





            



