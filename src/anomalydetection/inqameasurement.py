import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from functools import partial
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm

class QFeatureMap_rff():

  def __init__(
          self,
          input_dim: int,
          dim: int = 100,
          gamma: float = 1,
          random_state=None,
          **kwargs
  ):
      super().__init__(**kwargs)
      self.input_dim = input_dim
      self.dim = dim
      self.gamma = gamma
      self.random_state = random_state
      self.vmap_compute = jax.jit(jax.vmap(self.compute, in_axes=(0, None, None, None), out_axes=0))

  
  def build(self):
    rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
    x = np.zeros(shape=(1, self.input_dim))
    rbf_sampler.fit(x)

    self.rbf_sampler = rbf_sampler
    self.weights = jnp.array(rbf_sampler.random_weights_)
    self.offset = jnp.array(rbf_sampler.random_offset_)
    self.dim = rbf_sampler.get_params()['n_components']

  def update_rff(self, weights, offset):
    self.weights = jnp.array(weights)
    self.offset = jnp.array(offset)


  def get_dim(self, num_features):
    return self.dim

  @staticmethod
  def compute(X, weights, offset, dim):
    vals = jnp.dot(X, weights) + offset
    #vals = jnp.einsum('i,ik->k', X, weights) + offset
    vals = jnp.cos(vals)
    vals *= jnp.sqrt(2.) / jnp.sqrt(dim)
    return vals
    
  @partial(jit, static_argnums=(0,))
  def __call__(self, X):
    vals = self.vmap_compute(X, self.weights, self.offset, self.dim)
    norms = jnp.linalg.norm(vals, axis=1)
    psi = vals / norms[:, jnp.newaxis]
    return psi   



class InqaMeasurement():
  def __init__(self, input_shape, dim_x, gamma, random_state=None, batch_size = 300, window_size=10):
    self.gamma = gamma
    self.dim_x = dim_x
    self.fm_x = QFeatureMap_rff( input_dim=input_shape, dim = dim_x, gamma = gamma, random_state = random_state)
    self.fm_x.build()
    self.num_samples = 0 
    self.train_pure_batch = jax.jit(jax.vmap(self.train_pure, in_axes=(0)))
    self.collapse_batch = jax.jit(jax.vmap(self.collapse, in_axes=(0, None)))
    self.sum_batch = jax.jit(self.sum)
    self.key = jax.random.PRNGKey(random_state)
    self.batch_size = batch_size
    self.window_size = window_size
    self.inner_memory = None
    self.rho_res = None

  @staticmethod
  def train_pure(inputs):
    oper = jnp.einsum(
        '...i,...j->...ij',
        inputs, jnp.conj(inputs),
        optimize='optimal') # shape (b, nx, nx)
    return oper

  @staticmethod
  def sum(rho_res):
    return jnp.sum(rho_res, axis=0) 

  @staticmethod
  def verify_memory(batch, window_size, inner_memory):
      if inner_memory == None:
          inner_memory = batch[-window_size:, :]
      else:
          inner_memory = jnp.concatenate([inner_memory, batch], axis=0)[-window_size:]
      return inner_memory 


  @staticmethod
  @partial(jit, static_argnums=(1,2,3,4))
  def compute_training_jit(batch, window_size, fm_x, train_pure_batch, sum_batch,  inner_memory):
      updated_memory = InqaMeasurement.verify_memory(batch, window_size, inner_memory)  

      inputs = fm_x(updated_memory)
      rho_res = train_pure_batch(inputs)
      return sum_batch(rho_res), updated_memory

  @staticmethod
  def compute_training(values, window_size, perm, i, batch_size, fm_x, train_pure_batch,
                       sum_batch, compute_training_jit, inner_memory):
      batch_idx = perm[i * batch_size: (i + 1)*batch_size]
      batch = values[batch_idx, :]
      return compute_training_jit(batch, window_size, fm_x, train_pure_batch, sum_batch, inner_memory)

  def initial_train(self, values):
    num_batches = InqaMeasurement.obtain_params_batches(values, self.batch_size,  self.key)
    num_train = values.shape[0]
    perm = jnp.arange(num_train)
    for i in range(num_batches):
      if self.rho_res == None:
        self.rho_res, self.inner_memory = self.compute_training(values, self.window_size, perm, i, self.batch_size, self.fm_x, 
                                             self.train_pure_batch, self.sum_batch, self.compute_training_jit, 
                                             self.inner_memory)
      else:
        self.rho_res, self.inner_memory = self.compute_training(values, self.window_size, perm, i, self.batch_size, self.fm_x,
                                             self.train_pure_batch, self.sum_batch, self.compute_training_jit,
                                            self.inner_memory)
    self.num_samples += values.shape[0]  

    

  @staticmethod
  def collapse(inputs, rho_res):
    rho_h = jnp.matmul(jnp.conj(inputs), rho_res)
    rho_res = jnp.einsum(
        '...i, ...i -> ...',
        rho_h, jnp.conj(rho_h), 
        optimize='optimal') # shape (b,)
    return rho_res

  @staticmethod
  def obtain_params_batches(values, batch_size, key):
    num_train = values.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

  

  @partial(jit, static_argnums=(0,))
  def predict(self, values):
    if(self.rho_res == None or self.inner_memory == None):
        raise Exception("The model have to be trained before using it to predict.")

    num_batches = InqaMeasurement.obtain_params_batches(values, self.batch_size, self.key)
    results = None
    rho_res = self.rho_res / self.window_size
    num_train = values.shape[0]
    perm = jnp.arange(num_train)
    for i in range(num_batches):
      batch_idx = perm[i * self.batch_size: (i + 1)*self.batch_size]
      batch = values[batch_idx, :]

      inputs = self.fm_x(batch)
      batch_probs = self.collapse_batch(inputs, rho_res)
      results = jnp.concatenate([results, batch_probs], axis=0) if results is not None else batch_probs
    return results

