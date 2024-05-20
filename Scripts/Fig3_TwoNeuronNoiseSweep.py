'''
This is the two neuron mechanisms with rewards [0., 0.5]. It does a sweep over a small
set of sensory & motor activity and weight noise for long term distributions.

100 seeds per trial.
500,000 weight update steps per simulation.
1 activity settling step.
0.01 activity and weight learning rates.

Input parameter labels in the output files:

Files are saved as "logs_etaa{label}_etaw{label}_s{label}.pkl"


SLURM PARAMETERS:
Runtime for 1000k steps is X mins X secs. 
Memory for saving levers only is X MB.
'''

import joblib
import sys

import numpy as np
from copy import deepcopy

import jax.numpy as jnp
from jax import grad, jit
from jax import random
import jax

from IPython.display import clear_output

sqsum = lambda x: jnp.sum(jnp.square(x))


'''''''''''''''''''''''''''''''''''''''''''''''''''
Functions unchanged from from pan.py
'''''''''''''''''''''''''''''''''''''''''''''''''''

### Init #######################################################################

def init_params(hps) -> list:
  '''Returns arrays of initial activities and weights.
  Also returns a key for random generation.
  Inputs:
  Layer sizes and random seed.
  Init scale is how widely distributed the weights start out.
  '''

  def init_weights(sizes: list, key, scale_factor) -> list:
    keys = random.split(key, num=len(sizes))
    return [jnp.array(random_layer_params(m, n, k, scale_factor)) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

  def random_layer_params(m: int, n: int, key, scale_factor=1.):
    '''Returns a jax array of random numbers in (n, m) shape.
    This version is He initialization.
    '''
    scale = jnp.sqrt(2/m) * scale_factor
    return scale * random.normal(key, (n, m))

  activities = [jnp.zeros(s) for s in hps['sizes']]
  key = random.PRNGKey(hps['seed'])
  key, subkey = random.split(key)
  weights = init_weights(hps['sizes'], subkey, hps['init_scale'])

  return activities, weights, key



### Noise and clip functions ##########################################################################

@jit
def weight_noise(weights, key, hps):
  '''Adds some noise to the weights.
  '''
  new_weights = [[] for _ in weights]
  for l in range(len(weights)):
    key, subkey = random.split(key)
    noise = random.normal(subkey, weights[l].shape) * hps['eta_w']
    new_weights[l] = weights[l] + noise

  return new_weights, key

@jit
def weight_clip(weights, cap=2.):
  '''Makes sure weights don't go above some magnitude.
  '''
  new_weights = [[] for _ in weights]
  for l in range(len(weights)):
    new_weights[l] = jnp.clip(weights[l], -cap, cap)

  return new_weights


'''''''''''''''''''''''''''''''''''''''''''''''''''
-------------------- Update functions
'''''''''''''''''''''''''''''''''''''''''''''''''''

@jit
def update_acts(inp, activities, weights, hps, grad_clip=10., lr_factor=1.):
  act_grads = grad(total_loss, argnums=1)(inp, activities, weights, hps)
  return [act - hps['alpha'] * lr_factor * jnp.clip(d_act, -grad_clip, grad_clip) for act, d_act in zip(activities, act_grads)]

@jit
def update_weights(inp, activities, weights, hps, grad_clip=10., lr_factor=1.):
  w_grads = grad(total_loss, argnums=2)(inp, activities, weights, hps)
  return [w - hps['omega'] * lr_factor * jnp.clip(d_w, -grad_clip, grad_clip) for w, d_w in zip(weights, w_grads)]


  ### Log ##########################################################################

class Log():
  ''' Records all current variables in the network.
  .acts[i][t][n] is a list of activities for the ith layer, timestep t, neuron n.
  '''
  def __init__(self, hps):

    self.acts = [[] for _ in range(len(hps['sizes']))]
    self.weights = [[] for _ in range(len(hps['sizes'])-1)]
    self.actions = []
    self.energy = []

  def record(self, activities, weights, action):
    [self.acts[i].append(acts) for i,acts in enumerate(activities)]
    [self.weights[i].append(weights) for i,weights in enumerate(weights)]
    self.actions.append(action)

  def record_energy(self, loss_fun, inp, acts, weights, hps):
    self.energy.append(loss_fun(inp, acts, weights, hps))

  def close(self):
    # Loop over all logs and turn them into workable numpy arrays
    attributes_dict = vars(self)
    for attribute, values in attributes_dict.items():
      if attribute=='actions':
        self.actions = np.array(self.actions)
      else:
        setattr(self, attribute, [np.array(x) for x in values])


'''''''''''''''''''''''''''''''''''''''''''''''''''
----------- Mask function for connectivity tests
'''''''''''''''''''''''''''''''''''''''''''''''''''

# NOTE: unused, but here for consistency with pan.py so that we do not have to change more functions.

def zero_weights(weights, hps):
    '''Zero weights that are False in the mask and cut their gradients.
    Mask is a list the same shape as weights.
    Mask required to make it jit-compatible.
    weights[l] shape is [outputs, inputs].

    To make the mask (example):
    mask = [jnp.ones_like(w, dtype=bool) for w in weights]
    mask[0] = mask[0].at[nconns:, 0].set(False)
    '''


    if 'mask' not in hps:
        return weights

    else:
      mask = hps['mask']
      # Redefine connections
      for l in range(len(weights)):
          weights[l] = jnp.where(mask[l], weights[l], jax.lax.stop_gradient(0.))

      return weights
      



'''''''''''''''''''''''''''''''''''''''''''''''''''
Functions CHANGED from from pan.py
'''''''''''''''''''''''''''''''''''''''''''''''''''


'''''''''''''''''''''''''''''''''''''''''''''''''''
-------------------- Loss functions
'''''''''''''''''''''''''''''''''''''''''''''''''''

@jit
def linear_pred_loss(inp, acts, weights, hps):
  """CHANGES: no relu, no stopgrad.
  """

  predloss = 0

  # Add loss from not matching inp
  predloss += sqsum(acts[0]  - inp)

  # Now add losses from all layers
  for l in range(len(acts)-1):
    predloss += sqsum(acts[l+1] - jnp.matmul(weights[l], acts[l]))

  return predloss


@jit
def total_loss(inp, acts, weights, hps):
  '''CHANGES: uses linear_pred_loss
  '''

  # Allows for architecture specification via mask, if 'mask' is in hps
  weights = zero_weights(weights, hps)

  loss = linear_pred_loss(inp, acts, weights, hps)
  return loss


  '''''''''''''''''''''''''''''''''''''''''''''''''''
----------- Bandit env
'''''''''''''''''''''''''''''''''''''''''''''''''''

def two_neuron_bandit(motor, reward, key):
    '''Simple two neuron bandit. If motor<0, select lever 0 with no reward.
       If motor>0, select reward lever. Random choice for motor==0 for consistency
       with larger bandit task.
    '''
    
    # If tie, select random lever
    if motor == 0:
        lever = jax.random.choice(key, jnp.array([0,1]))
        new_key, _ = jax.random.split(key)
        return jnp.array([lever * reward]), lever, new_key
    # Else, simple bandit
    if motor < 0:
        return jnp.array([0.]), 0, key
    else:
        return jnp.array([reward]), 1, key


 ### Noise and clip functions ##########################################################################

@jit
def act_noise(activities, key, hps, only_sensory_noise=False):
  '''CHANGES: allows for only sensory noise.
  '''

  noise_scale = hps['eta_a']

  new_activities = [[] for _ in activities]
  for l in range(len(activities)):

    key, subkey = random.split(key)
    noise = random.normal(subkey, activities[l].shape) * noise_scale
    new_activities[l] = activities[l] + noise * jnp.where(l > 0, 1-only_sensory_noise, True)

  return new_activities, key


'''''''''''''''''''''''''''''''''''''''''''''''''''
----------- Running sims
'''''''''''''''''''''''''''''''''''''''''''''''''''

def run_PaN(hps, rewards, timesteps, only_sensory_noise=False, settle_time=1, init_log=None, init_seed=None):
    '''CHANGES: default settle_time=1. bandit() replaced with two_neuron_bandit().
                allows for only sensory noise.
    '''

    # Initialize network. If there's an init_log, load weights and activities from there.
    # Weights are random; acts are 0
    acts, weights, key = init_params(hps)

    if init_log is not None:
        acts = [jnp.array(init_log.acts[i][-1]) for i in range(len(init_log.acts))]
        weights = [jnp.array(init_log.weights[i][-1]) for i in range(len(init_log.weights))]
        key = random.PRNGKey(init_seed)

    # Initialize loggers
    log = Log(hps)

    ''' Simulation begins '''
    for t in range(timesteps):
        if t%10_000==0:
          clear_output(wait=True)
          print('Timestep', t)

        # Get outputs for bandit
        signal, lever, key = two_neuron_bandit(acts[-1], rewards, key)

        # Activities settle
        for j in range(settle_time):

          # Update activities
          acts = update_acts(signal, acts, weights, hps)
          # Add noise
          acts, key = act_noise(acts, key, hps, only_sensory_noise)


        # Weight update
        weights = update_weights(signal, acts, weights, hps)
        # Add noise
        weights, key = weight_noise(weights, key, hps)
        weights = weight_clip(weights)


        # Log
        log.record(acts, weights, lever)

    # Make objects in log easier to work with
    log.close()

    clear_output(wait=True)
    print('Done')
    return log


'''''''''''''''''''''''''''''''''''''''''''''''''''
-------------------- Main
'''''''''''''''''''''''''''''''''''''''''''''''''''
def main():

    ''' Set parameters '''
    # Get inputs and set parameters --------------------------------
    iteration_index = int(sys.argv[1])

    # Set parameter combinations
    eta_a = [0., 0.0001, 0.00024, 0.00056, 0.0013, 0.0032, 0.0075, 0.018, 0.042, 0.1]
    eta_w = eta_a
    seeds = [8924 + s * 13 for s in range(100)]

    parameter_combinations = [
	    {'eta_a': a, 'eta_w': w, 'seed': seed}
	    for a in eta_a
	    for w in eta_w
	    for seed in seeds]

	# Check that iteration index is valid
    if iteration_index < 0 or iteration_index >= len(parameter_combinations):
        print("Error: Invalid iteration index.")
        sys.exit(1)

    # Choose the parameter combination based on the iteration index
    chosen_parameters = parameter_combinations[iteration_index]
    etaa, etaw, s = chosen_parameters["eta_a"], chosen_parameters["eta_w"], chosen_parameters["seed"]

    # Set filename
    fname = f"log_etaa{etaa}_etaw{etaw}_s{s}.pkl"



    ''' The simulation bit! '''

 	# Set environmental feedback options
    reward = jnp.array([0.5])

	# Set hyperparameters (empirically chosen)
    timesteps = 500_000

    hps = {

	  'seed'       : s,
	  'sizes'      : [1, 1], # Number of neurons in each layer
	  'init_scale' : 1,         # Width of weight init distribution; default He

	  # Learning parameters
	  'alpha'  : 0.01, # Activity update rate
	  'omega'  : 0.01, # Weight update rate

	  # Network properties
	  'eta_a'  : etaa, # Activity noise scale
	  'eta_w'  : etaw, # Weight noise scale
	}

    log = run_PaN(hps, reward, timesteps)


    ''' Save data '''
    joblib.dump(log, fname, compress=('zlib', 3))


# Run main()
if __name__=='__main__':
  main()

