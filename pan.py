'''
Base functions
'''

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
----------- Mask function for connectivity tests
'''''''''''''''''''''''''''''''''''''''''''''''''''

@jit
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
-------------------- Losses and update functions
'''''''''''''''''''''''''''''''''''''''''''''''''''

@jit
def pred_loss(stimuli, acts, weights, hps):
  '''Calculates overall prediction loss for not-fixed-output case (input still determined by env.)
  '''

  predloss = 0

  # Add loss from not matching input stimuli
  predloss += sqsum( acts[0]  - relu(stimuli) )

  # Now add loss from first layer without it affecting input layer
  act0_copy = jax.lax.stop_gradient(deepcopy(acts[0]))
  predloss += sqsum( acts[1] - relu(jnp.matmul(weights[0], act0_copy) ) )

  # Now add losses from all layers
  for l in range(len(acts)-2):
    predloss += sqsum( acts[l+2] - relu(jnp.matmul(weights[l+1], acts[l+1])) )

  return predloss

@jit
def total_loss(inp, acts, weights, hps):
  '''Sums up loss terms in here. In this script that's only predictive loss
  '''

  # Allows for architecture specification via mask, if 'mask' is in hps
  weights = zero_weights(weights, hps)

  loss = pred_loss(inp, acts, weights, hps)
  return loss

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
