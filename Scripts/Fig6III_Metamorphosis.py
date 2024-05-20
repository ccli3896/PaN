'''''''''''''''''''''''''''''''''''''''''''''

3 peak landscape.
1000 seeds, 500,000 steps per run.

Input parameter labels in the output files:

  random seed:
    range(1000)

SLURM PARAMETERS:
Runtime for 10k steps, 00:01:00 (hr:min:sec) on local machine ish.
Cluster CPU time is roughly 3.33x slower. Allocated 8 h just in case.
Memory for saving levers only for 10k run is 20 kb. Allocated 10GB across 1000 jobs.

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

  # Now add losses from all layers
  for l in range(len(acts)-1):
    predloss += sqsum( (acts[l+1] - relu(jnp.matmul(weights[l], acts[l]))) )

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
-------------------- Landscape fns
'''''''''''''''''''''''''''''''''''''''''''''''''''

def landscape_generator(peak_value, peak_ind):
    def generate_pyramid(i, j):
        # Makes the pyramid on the fly so that it's not constantly reading from an enormous array.
        # m is the size of the whole object.

        m = 4000
        peak_location = (1000,1000)
        
        if i >= m or j >= m:
            raise ValueError('Indices out of bounds.')

        if peak_ind==1:
            i = 4000-i
        elif peak_ind==2:
            j = 4000-j

        peak_x, peak_y = peak_location
        return min((i)*peak_value/peak_y, 
                   (2*peak_y-(i)) * (peak_value/peak_y),
                   (j)*peak_value/peak_x,
                   (2*peak_x-(j)) * (peak_value/peak_x))
        
    return generate_pyramid

'''Landscape functions'''
# Note: ALL LOCS ARE (Y,X) because of matrix indexing. I think this is less confusing

def rewards_from_landscape_rot(loc, rot, landscape, landscape_size):
    '''GPT. Takes a location (index tuple) and finds the four surrounding reward values in 2D landscape,
    written as a function.
    Returns them as an array, a set of rewards for levers, to be used in bandit().
    '''
    i, j = loc
    n_rows, n_cols = landscape_size, landscape_size

    if rot==0: # Right
        neighbor = landscape(i, (j+1) % n_cols)  # Right neighbor
    elif rot==1: # Up
        neighbor = landscape((i-1) % n_rows, j)  # Upper neighbor
    elif rot==2: # Left
        neighbor = landscape(i, (j-1) % n_cols)  # Left neighbor
    elif rot==3: # Down
        neighbor = landscape((i+1) % n_rows, j)  # Lower neighbor
    else:
        raise ValueError('Invalid rotation')

    return jnp.array([
        neighbor,
        landscape(i,j),    # Stay
        landscape(i,j),    # Stay 
    ]) - landscape(i,j)

        

def move_in_landscape_rot(action, loc, rotation, landscape_size):
    '''GPT. Takes an action in [0, 1, 2] that corresponds to forward, turn CCW, turn CW.
    Returns new location. 
    Rotations are, in order: 0 deg (right), 90 deg (up), 180 deg, 270 deg.
    Wraparound is 0 for False, 1 for True because I can't figure out how to jit bools
    '''
    i,j = loc
    n_rows, n_cols = landscape_size, landscape_size

    # Implement controls with wrapping around borders
    if action == 0:  # Forward
        if rotation == 0:
            j = (j + 1) % n_cols # Right
        elif rotation == 1:
            i = (i - 1) % n_rows # Up
        elif rotation == 2:
            j = (j - 1) % n_cols # Left
        elif rotation == 3:
            i = (i + 1) % n_rows # Down
        else:
            raise ValueError('Invalid rotation')

    # Just rotate
    elif action == 1:
        rotation = (rotation + 1) % 4 # CCW rotation
    elif action == 2:
        rotation = (rotation - 1) % 4 # CW rotation
    else:
        raise ValueError('Invalid action')
         
    return (i,j), rotation

def init_loc_rot(landscape_size, key):
    # Randomly initialize a location and rotation [0,3] in landscape. randint documentation: [minval, maxval).
    # Get bounds of landscape
    n_rows, n_cols = landscape_size, landscape_size
    
    key, sub0, sub1, sub2 = random.split(key, 4)
    j = jax.random.randint(sub0, (1,), minval=0, maxval=n_cols)
    i = jax.random.randint(sub1, (1,), minval=0, maxval=n_rows)
    r = jax.random.choice(sub2, 4)
    return jnp.array([i,j]).flatten(), r, key

# Translational functions
def rewards_from_landscape(loc, landscape, landscape_size):
    '''GPT. Takes a location (index tuple) and finds the four surrounding reward values in 2D landscape,
    written as a function.
    Returns them as an array, a set of rewards for levers, to be used in bandit().
    '''
    i, j = loc
    n_rows, n_cols = landscape_size, landscape_size

    return jnp.array([
        landscape((i-1) % n_rows, j),  # Upper neighbor
        landscape((i+1) % n_rows, j),  # Lower neighbor
        landscape(i, (j-1) % n_cols),  # Left neighbor
        landscape(i, (j+1) % n_cols),  # Right neighbor
        landscape(i,j)                 # Stay (possible option, not used here)
    ]) - landscape(i,j)

        

def move_in_landscape(action, loc, landscape_size):
    '''GPT. Takes an action in [0,1,2,3] that corresponds to up, down, left, right (UDLR). 
    Returns new location. 
    Wraparound is 0 for False, 1 for True because I can't figure out how to jit bools
    '''
    i,j = loc
    n_rows, n_cols = landscape_size, landscape_size

    # Implement controls with wrapping around borders
    if action == 0:  # Up
        i = (i - 1) % n_rows
    elif action == 1:  # Down
        i = (i + 1) % n_rows
    elif action == 2:  # Left
        j = (j - 1) % n_cols
    elif action == 3:  # Right
        j = (j + 1) % n_cols
        
    return i,j


'''''''''''''''''''''''''''''''''''''''''''''''''''
-------------------- Main
'''''''''''''''''''''''''''''''''''''''''''''''''''

def main():

  # Get inputs and set parameters --------------------------------
  s = int(sys.argv[1])
  
  # Check if the iteration index is valid
  if s >= 1000:
      print("Error: Invalid iteration index.")
      sys.exit(1)

  # Set filename
  fname = f"landscape_Metamorphosis_s{s}.pkl"


  # Actual simulation bit ---------------------------------------

  # Set hyperparameters
  timesteps = 500_000
  settle_time = 10

  hps = {

    'seed'  : 92 + 15*s,
    'sizes' : [1, 30, 3],

    # Learning parameters
    'alpha'  : 0.01, # Activity update rate
    'omega'  : 0.01, # Weight update rate

    # Network properties
    'eta_a'  : 0.01, # Activity noise scale
    'eta_w'  : 0.0001, # Weight noise scale
  }

  # Set landscapes as functions
  landscape_size = 4000
  gp0 = landscape_generator(500, 0)
  gp1 = landscape_generator(750, 1)
  gp2 = landscape_generator(1000, 2)
  gps = lambda i,j: jnp.maximum(jnp.maximum(jnp.maximum(gp0(i,j), gp1(i,j)), gp2(i,j)), 0.)

  landscape = gps

  # Initialize all five network states. Weights are random; acts are 0
  acts, weights, key = init_params(hps)
  loc, rot, key = init_loc_rot(landscape_size, key)

  # Initialize logger
  locs = []


  ''' Simulation begins '''
  ''' FIRST STAGE: ROTATIONAL MOVEMENT '''
  for t in range(timesteps):
    if t%100_000==0:
      print(t)

    # Record location and get rewards from landscape
    locs.append(loc)
    rewards = rewards_from_landscape_rot(loc, rot, landscape, landscape_size)
      
    # Get outputs for bandit
    reward, lever, key = bandit(acts[-1], rewards, key)
    # Set stimuli 
    stimuli = [reward]
      
    # Move in landscape
    loc, rot = move_in_landscape_rot(lever, loc, rot, landscape_size)
    
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



  # Modify motor stuff--add a neuron
  hps['sizes'] = [1, 30, 4]

  _, temp_weights, _ = init_params(hps)
  weights[1] = temp_weights[1].at[:3, :].set(weights[1])
  acts[2] = jnp.concatenate([acts[2], jnp.array([0.])])



  ''' SECOND STAGE: TRANSLATIONAL MOVEMENT '''
  for t in range(timesteps):
    if t%100_000==0:
      print(t)

    # Record location and get rewards from landscape
    locs.append(loc)
    rewards = rewards_from_landscape(loc, landscape, landscape_size)
      
    # Get outputs for bandit
    reward, lever, key = bandit(acts[-1], rewards, key)
    # Set stimuli 
    stimuli = [reward]
      
    # Move in landscape
    loc = move_in_landscape(lever, loc, landscape_size)
    
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


  # Save lever choices
  joblib.dump(np.array(locs), fname, compress=('zlib', 3))


# Run main()
if __name__=='__main__':
  main()
