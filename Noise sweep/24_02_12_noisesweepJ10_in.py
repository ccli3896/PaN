'''
This is the usual 3-arm bandit script with rewards [0, 0, 0.5]. It does a noise sweep
over activities and weights on a log scale from 10^-4 to 10^-1.

25 seeds per trial.
500,000 weight update steps per simulation.
10 activity settling steps based on past parameter searches. NOTE noise is added INSIDE the loop.
0.01 activity and weight learning rates.

Input parameter labels in the output files:

Files are saved as "levers_etaa{label}_etaw{label}_s{label}_j10.pkl"

  eta_(both):  
    label    0    1        2         3         4         5         6         7         8        9
    value    0.   0.0001   0.00024   0.00056   0.0013    0.0032    0.0075    0.018     0.042    0.1

  random seed:
    8924 + s * 13 for s in range(25)


SLURM PARAMETERS:
Runtime for 500k steps is ~ 12 mins 30 secs. Give it an hour on the cluster
Memory for saving levers only is 10 kb. Giving it 5 GB total in case (?)
'''


import joblib
import sys
import itertools

import numpy as np
from copy import deepcopy

import jax.numpy as jnp
from jax import grad, jit
from jax import random
import jax

sqsum = lambda x: jnp.sum(jnp.square(x))
relu = lambda x: jnp.maximum(0, x)

### Init #######################################################################

def init_params(hps) -> list:
  '''Returns arrays of initial activities and weights.
  Also returns a key for random generation.
  Inputs:
  Layer sizes and random seed.
  '''

  def init_weights(sizes: list, key) -> list:
    keys = random.split(key, num=len(sizes))
    return [jnp.array(random_layer_params(m, n, k)) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

  def random_layer_params(m: int, n: int, key):
    '''Returns a jax array of random numbers in (n, m) shape.
    This version is He initialization.
    '''
    scale = jnp.sqrt(2/m)
    return scale * random.normal(key, (n, m))

  activities = [jnp.zeros(s) for s in hps['sizes']]
  key = random.PRNGKey(hps['seed'])
  key, subkey = random.split(key)
  weights = init_weights(hps['sizes'], subkey)

  return activities, weights, key

### Noise and clip functions ##########################################################################

@jit
def act_noise(activities, key, hps):
  ''' Adds noise to each neuron.
  '''

  noise_scale = hps['eta_a']

  new_activities = [[] for _ in activities]
  for l in range(len(activities)):

    key, subkey = random.split(key)
    noise = random.normal(subkey, activities[l].shape) * noise_scale
    new_activities[l] = activities[l] + noise

  return new_activities, key

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
-------------------- Losses and update functions
'''''''''''''''''''''''''''''''''''''''''''''''''''

@jit
def pred_loss(stimuli, acts, weights, hps):
  ''' Calculates overall prediction loss for not-fixed-output case (input still determined by env.)
  Stimuli variable is list of len 1 (to be consistent with target prop pred_loss function). The single 
  element in there is the sensory input as a jax array.
  '''

  predloss = 0

  # Add loss from not matching input stimuli
  predloss += sqsum( (acts[0]  - relu(stimuli[0])) )

  # Now add losses from all layers
  for l in range(len(acts)-1):
    predloss += sqsum( (acts[l+1] - relu(jnp.matmul(weights[l], acts[l]))) )

  return predloss

@jit
def total_loss(stimuli, acts, weights, hps):
  ''' Sums up loss terms in here. In this script that's only predictive loss
  '''
  loss = pred_loss(stimuli, acts, weights, hps)
  return loss

@jit
def update_acts(stimuli, activities, weights, hps, grad_clip=10.):
  act_grads = grad(total_loss, argnums=1)(stimuli, activities, weights, hps)
  return [act - hps['alpha'] * jnp.clip(d_act, -grad_clip, grad_clip) for act, d_act in zip(activities, act_grads)]

@jit
def update_weights(stimuli, activities, weights, hps, grad_clip=10.):
  w_grads = grad(total_loss, argnums=2)(stimuli, activities, weights, hps)
  return [w - hps['omega'] * jnp.clip(d_w, -grad_clip, grad_clip) for w, d_w in zip(weights, w_grads)]


'''''''''''''''''''''''''''''''''''''''''''''''''''
-------------------- Bandit
'''''''''''''''''''''''''''''''''''''''''''''''''''

def bandit(motors, rewards, key):
    '''Takes a set of motor outputs from network and finds argmax. 
    Returns corresponding reward from reward vector.
    Can't be jitted for some reason but oh well
    '''
    
    def argmax_tiebreaker(arr, rng_key):
        '''Does argmax but returns random choice between equal maxima.
        Thank you chatGPT
        '''
        max_indices = jnp.where(arr == arr.max())[0]
        chosen_index = jax.random.choice(rng_key, max_indices)
        new_rng_key, _ = jax.random.split(rng_key)
        return chosen_index, new_rng_key
        
    lever_ind, key = argmax_tiebreaker(motors, key)
    reward = jnp.array(rewards[lever_ind])
    return reward, lever_ind, key


'''''''''''''''''''''''''''''''''''''''''''''''''''
-------------------- Main
'''''''''''''''''''''''''''''''''''''''''''''''''''

def main():

  # Get inputs and set parameters --------------------------------
  iteration_index = int(sys.argv[1])
  
  # Parameter values
  eta_values = [0., 0.0001, 0.00024, 0.00056, 0.0013, 0.0032, 0.0075, 0.018, 0.042, 0.1]
  random_seed_range = range(25)

  # Create the Cartesian product of parameter sets
  parameter_combinations = list(itertools.product(eta_values, eta_values, random_seed_range))
  
  # Check if the iteration index is valid
  if iteration_index < 0 or iteration_index >= len(parameter_combinations):
      print("Error: Invalid iteration index.")
      sys.exit(1)
  
  # Choose the parameter combination based on the iteration index
  chosen_parameters = parameter_combinations[iteration_index]
  etaa, etaw, s = chosen_parameters

  # Get the indices of each parameter in the original lists for filename
  etaa_index, etaw_index, random_seed_index = [lst.index(chosen_parameters[i]) for i, lst in enumerate([eta_values, eta_values, list(random_seed_range)])]

  # Set filename
  fname = f"levers_etaa{etaa_index}_etaw{etaw_index}_s{random_seed_index}_j10_IN.pkl"



  # Actual simulation bit ---------------------------------------

  # Set hyperparameters
  timesteps = 500_000

  # Set bandit rewards
  rewards = jnp.array([0., 0., .5])

  hps = {

      'seed'  : 8924 + s * 13,
      'sizes' : [1, 30, 3],

      # Learning parameters
      'alpha'  : 0.01, # Activity update rate
      'omega'  : 0.01, # Weight update rate

      # Network properties
      'eta_a'  : etaa, # Activity noise scale
      'eta_w'  : etaw, # Weight noise scale
  }


  # Initialize all five network states. Weights are random; acts are 0
  acts, weights, key = init_params(hps)

  # Initialize logger
  levers = []

  ''' Simulation begins '''

  for t in range(timesteps):
      if t%100_000==0:
        print(t)

      # Get outputs for bandit
      reward, lever, key = bandit(acts[-1], rewards, key)
      # Set stimuli 
      stimuli = [reward]
      # Record lever choice
      levers.append(lever)
      
      for j in range(10):
        # Update activities
        acts = update_acts(stimuli, acts, weights, hps)
        # Add noise
        acts, key = act_noise(acts, key, hps)
      
      # Weight update
      weights = update_weights(stimuli, acts, weights, hps)
      # Add noise
      weights, key = weight_noise(weights, key, hps)
      # Clip weights
      weights = weight_clip(weights)


  # Save lever choices
  levers_arr = np.array(levers)
  joblib.dump(levers_arr, fname, compress=('zlib', 3))



# Run main()
if __name__=='__main__':
  main()