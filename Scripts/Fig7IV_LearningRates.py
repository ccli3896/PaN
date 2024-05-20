'''
Base functions
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
  '''Calculates overall prediction loss for not-fixed-output case (input still determined by env.)
  Stimuli variable is list of len 1 (to be consistent with target prop pred_loss function). The single 
  element in there is the sensory input as a jax array.
  '''

  predloss = 0

  # Add loss from not matching input stimuli
  predloss += sqsum( (acts[0]  - relu(stimuli[0])) )


  # Now add loss from first layer without it affecting input layer
  act0_copy = jax.lax.stop_gradient(deepcopy(acts[0]))
  predloss += sqsum( (acts[1] - relu(jnp.matmul(weights[0], act0_copy))) )

  # Now add losses from all layers
  for l in range(len(acts)-1):
    predloss += sqsum( (acts[l+1] - relu(jnp.matmul(weights[l], acts[l]))) )

  return predloss

@jit
def total_loss(inp, acts, weights, hps):
  '''Sums up loss terms in here. In this script that's only predictive loss
  '''

  # # Allows for architecture specification via mask, if 'mask' is in hps
  # weights = zero_weights(weights, hps)

  loss = pred_loss(inp, acts, weights, hps)
  return loss


@jit
def update_acts(stimuli, activities, weights, hps, grad_clip=10.):
  act_grads = grad(total_loss, argnums=1)(stimuli, activities, weights, hps)
  return [act - hps['alpha'] * jnp.clip(d_act, -grad_clip, grad_clip) for act, d_act in zip(activities, act_grads)]

@jit
def update_weights(stimuli, activities, weights, hps, grad_clip=10.):
  w_grads = grad(total_loss, argnums=2)(stimuli, activities, weights, hps)
  return [w - hps['omega'] * jnp.clip(d_w, -grad_clip, grad_clip) for w, d_w in zip(weights, w_grads)]



### Log ##########################################################################

class Log():
  ''' Records all current variables in the network.
  .acts[i][t][n] is a list of activities for the ith layer, timestep t, neuron n.
  '''
  def __init__(self, hps):

    self.acts = [[] for _ in range(len(hps['sizes']))]
    self.weights = [[] for _ in range(len(hps['sizes'])-1)]
    self.energy = []

  def record(self, activities, weights):
    [self.acts[i].append(acts) for i,acts in enumerate(activities)]
    [self.weights[i].append(weights) for i,weights in enumerate(weights)]

  def record_energy(self, loss_fun, stimuli, acts, weights, acts_ad, hps):
    self.energy.append(loss_fun(stimuli, acts, weights, acts_ad, hps))

  def close(self):
    # Loop over all logs and turn them into workable numpy arrays
    attributes_dict = vars(self)
    for attribute, values in attributes_dict.items():
      setattr(self, attribute, [np.array(x) for x in values])


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
-------------------- Actual simulation bit
'''''''''''''''''''''''''''''''''''''''''''''''''''


def main():


  # Get inputs and set parameters --------------------------------
  iteration_index = int(sys.argv[1])
  
  # Parameter values
  alpha_values = [0., 0.0001, 0.00024, 0.00056, 0.0013, 0.0032, 0.0075, 0.018, 0.042, 0.1]
  omega_values = [0., 0.0001, 0.00024, 0.00056, 0.0013, 0.0032, 0.0075, 0.018, 0.042, 0.1]  
  random_seed_range = list(range(25))

  # Create the Cartesian product of parameter sets
  parameter_combinations = list(itertools.product(alpha_values, omega_values, random_seed_range))
  
  # Check if the iteration index is valid
  if iteration_index < 0 or iteration_index >= len(parameter_combinations):
      print("Error: Invalid iteration index.")
      sys.exit(1)
  
  # Choose the parameter combination based on the iteration index
  chosen_parameters = parameter_combinations[iteration_index]
  alpha_da, omega_da, s = chosen_parameters

  # Get the indices of each parameter in the original lists for filename
  alpha_index, omega_index, random_seed_index = [lst.index(chosen_parameters[i]) for i, lst in enumerate([alpha_values, omega_values, random_seed_range])]

  # Set filename
  fname = f"_alphai{alpha_index}_omegai{omega_index}_s{random_seed_index}.pkl"


  # Run  --------------------------------

  rewards = jnp.array([[0.5, 0.], [0., 0.5], [0., 0.]])
  timesteps = 500_000
  settle_time = 10

  hps = {
    'seed'  : s,
    'sizes' : [2, 30, 3],
  
    # Learning parameters
    'alpha'  : 0.01, # Activity update rate
    'omega'  : 0.01, # Weight update rate
  
    # Network properties
    'eta_a'  : 0.01, # Activity noise scale
    'eta_w'  : 0.0001, # Weight noise scale
  }
  
  # Initialize logger
  levers = []

  # Initialize network states. Weights are random; acts are 0
  acts, weights, key = init_params(hps)

  

  ''' Simulation begins '''
  
  for t in range(timesteps):
      if t%100_000==0:
        print(t)
      
      # Get outputs for bandit
      stimuli, lever, key = bandit(acts[-1], rewards, key)

      # CHANGE LEARNING RATE
      if lever==0:
          hps['alpha'] = alpha_da
          hps['omega'] = omega_da
      else:
          hps['alpha'] = 0.01
          hps['omega'] = 0.01

      # Record lever choice
      levers.append(lever)
      
      # Activities settle
      for j in range(settle_time):
      
        # Update activities
        acts = update_acts(stimuli, acts, weights, hps)
        # Add noise
        acts, key = act_noise(acts, key, hps)
      
      # Weight update
      weights = update_weights(stimuli, acts, weights, hps)
      # Add noise and do weight adjustments
      weights, key = weight_noise(weights, key, hps)
      weights = weight_clip(weights)


  # Save outputs
  joblib.dump(np.array(levers), f'levers_LRBig{fname}', compress=('zlib', 3))


if __name__=='__main__':
  main()

