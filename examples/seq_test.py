import numpy as np
import matplotlib.pyplot as plt
from nbi import NBI, get_featurizer
from scipy.stats import uniform, truncnorm
from pathos.helpers import freeze_support

#Set random seed
np.random.seed(0)

'''
Example to do sequential inference with multiprocessing on any Windows OS

'''

#Model function
def sine(param):
    phi0, A, freq = param
    return np.sin(t * freq + phi0) * A

#Gaussian noise
def noise(x, y=None):
    """
    x: light curve of shape (length,)
    y: parameter of shape (dim,)
    """
    rand = np.random.normal(0, 1, size=x.shape[0])
    x_noise = x + rand * x_err
    if y is None:
        return x_noise
    else:
        return x_noise, y
    
#Log likelihood   
def log_like(x, x_path, params):
    # x is observed data, x_path is path to saved model prediction
    model =  np.load(x_path)
    chi2 = (((x - model) / x_err)**2).sum()
    return - chi2 / 2
	
# this is for sampling from the prior
def prior(n):
    """
    The first round training set is simulated from parameters drawn from the prior function below
    """
    phi0_prior = uniform(loc=0, scale=np.pi*2)
    A_prior = uniform(loc=1, scale=4)
    freq_prior = uniform(loc=2*np.pi, scale=10*np.pi)

    phi0 = phi0_prior.rvs(n)
    A = A_prior.rvs(n)
    freq = freq_prior.rvs(n)
    
    thetas = np.array([phi0, A, freq]).T
    return thetas

# this is for evaluating the prior probability
def log_prior(thetas):
    return 0	
	
#Create array of times
t = np.linspace(0,1,50)
#Gaussian noise of amplitude 0.5
x_err = 0.5

y_true = [1, 2, 4*np.pi]
x_obs = noise(sine(y_true))

labels = [r"$\phi_0$", r"$A$", r"$freq$"]

# hyperparameters for the normalizing flow
flow_config = {
    'flow_hidden': 32,
    'num_cond_inputs': 32,
    'num_blocks': 4,
    'perm_seed': 3,
    'n_mog': 1                # number of gaussian mixture
}

#Wrapper function to run the NBI engine
def seq_data():
    # nbi has pre-defined neural networks for sequential data
    resnet = get_featurizer('resnetrnn', 1, 32, depth=3)

    engine = NBI(
        resnet,
        dim_param=3,
        physics=sine,
        instrumental=noise,
        prior_sampler=prior,
        log_like=log_like,
        log_prior=log_prior,
        flow_config=flow_config,
        labels=labels,
        directory='test',
        n_jobs=10,         # for generating training set
        parallel=True,      # only useful if GPU available
        tqdm_notebook=True
    )


    engine.run(
        x_obs,
        y_true=y_true,
        n_rounds=2,
        n_per_round=5120,
        n_epochs=10,
        train_batch=64,
        val_batch=64,
        lr=0.0005,
        min_lr=0.0002,        # learning rate decays from lr to min_lr at the last epoch
        f_val=0.1,            # fraction used as validation set
        early_stop_patience=5 # stop training if val loss not improve by 5 epochs
    )
    
#This is necessary for MP to work on Windows    
if __name__ ==  '__main__': 
    #Use freeze_support to aid serialization across the process map
    freeze_support()
    #Run NBI
    seq_data()

