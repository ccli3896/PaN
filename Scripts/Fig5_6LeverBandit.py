'''''''''''''''''''''''''''''''''''''''''''''

6-armed bandit with rewards linearly spaced from 0 to 0.5. (increments of .1)
25 seeds per trial.

500,000 steps per run.

Input parameter labels in the output files:

  random seed:
    range(25)

Run locally

'''''''''''''''''''''''''''''''''''''''''''''

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
  for l in range(len(acts)-2):
    predloss += sqsum( (acts[l+2] - relu(jnp.matmul(weights[l+1], acts[l+1]))) )

  return predloss

@jit
def total_loss(stimuli, acts, weights, hps):
  '''Sums up loss terms in here. In this script that's only predictive loss
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
  s = int(sys.argv[1])
  
  # Check if the iteration index is valid
  if s >= 25:
      print("Error: Invalid iteration index.")
      sys.exit(1)

  # Set filename
  fname = f"levers_bigbandit_spaced_s{s}.pkl"


  # Actual simulation bit ---------------------------------------

  # Set hyperparameters
  timesteps = 500_000
  settle_time = 10

  # Set bandit rewards
  rewards = jnp.arange(0., 0.6, 0.1)

  hps = {

      'seed'  : 8924 + s * 13,
      'sizes' : [1, 30, len(rewards)],

      # Learning parameters
      'alpha'  : 0.01, # Activity update rate
      'omega'  : 0.01, # Weight update rate

      # Network properties
      'eta_a'  : 0.01, # Activity noise scale
      'eta_w'  : 0.0001, # Weight noise scale
  }


  # Initialize all five network states. Weights are random; acts are 0
  acts, weights, key = init_params(hps)

  # Initialize logger
  levers = []
  weightslog0 = []
  weightslog1 = []


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
      
      # Activities settle
      for j in range(settle_time):
      
        # Update activities
        acts = update_acts(stimuli, acts, weights, hps)
        # Add noise
        acts, key = act_noise(acts, key, hps)
      
      
      # Weight update
      weights = update_weights(stimuli, acts, weights, hps)
      # Add noise
      weights, key = weight_noise(weights, key, hps)
      weights = weight_clip(weights)

      weightslog0.append(weights[0])
      weightslog1.append(weights[1])


  # Save lever choices
  joblib.dump(np.array(levers), fname, compress=('zlib', 3))
  joblib.dump(np.array(weightslog0), f'bigbandit_spaced_weights0_s{s}.pkl', compress=('zlib', 3))
  joblib.dump(np.array(weightslog1), f'bigbandit_spaced_weights1_s{s}.pkl', compress=('zlib', 3))


# Run main()
if __name__=='__main__':
  main()