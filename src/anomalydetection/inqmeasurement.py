import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from functools import partial
from sklearn.kernel_approximation import RBFSampler


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



class InqMeasurement():
  def __init__(self, input_shape, dim_x, gamma, random_state=None, batch_size = 300):
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
  @partial(jit, static_argnums=(1,2,3,4))
  def compute_training_jit(batch, alpha, fm_x, train_pure_batch, sum_batch, rho):
      inputs = fm_x(batch)
      rho_res = train_pure_batch(inputs)
      rho_res = sum_batch(rho_res)
      return jnp.add((alpha)*rho_res, (1-alpha)*rho) if rho is not None else rho_res
      #return jnp.add(rho_res, rho) if rho is not None else rho_res

  @staticmethod
  def compute_training(values, alpha, perm, i, batch_size, fm_x, train_pure_batch, sum_batch, rho, compute_training_jit):
      batch_idx = perm[i * batch_size: (i + 1)*batch_size]
      batch = values[batch_idx, :]
      return compute_training_jit(batch, alpha, fm_x, train_pure_batch, sum_batch, rho)

  def initial_train(self, values, alpha):
    num_batches = InqMeasurement.obtain_params_batches(values, self.batch_size,  self.key)
    #print('Time obtain_params_batches: ', stop - start)  
    num_train = values.shape[0]
    perm = jnp.arange(num_train)
    for i in range(num_batches):
      #start = timeit.default_timer()
      #batch_idx = perm[i * self.batch_size: (i + 1)*self.batch_size]
      #stop = timeit.default_timer()
      #print('Time batch_idx: ', stop - start)
      #
      #start = timeit.default_timer()
      #batch = values[batch_idx, :]
      ##batch = values
      #stop = timeit.default_timer()
      #print('Time capture data: ', stop - start)
      #
      #
      #start = timeit.default_timer()
      #inputs = self.fm_x(batch)
      #stop = timeit.default_timer()
      #print('Time fm_x: ', stop - start)  
      #
      #start = timeit.default_timer()
      #rho_res = self.train_pure_batch(inputs)
      #stop = timeit.default_timer()
      #print('Time rho_res: ', stop - start)  
      #
      #start = timeit.default_timer()
      #rho_res = self.sum_batch(rho_res)
      #stop = timeit.default_timer()
      #print('Time sum rho_res: ', stop - start)  
      #print(self.fm_x.weights.shape)
      if hasattr(self, "rho_res"):
        self.rho_res = self.compute_training(values, alpha, perm, i, self.batch_size, self.fm_x, 
                                             self.train_pure_batch, self.sum_batch, self.rho_res, self.compute_training_jit)
      else:
        self.rho_res = self.compute_training(values, alpha, perm, i, self.batch_size, self.fm_x,
                                             self.train_pure_batch, self.sum_batch, None, self.compute_training_jit)
      #print('Time sum rho_res and self: ', stop - start)  
    self.num_samples += values.shape[0]  
    #print('Time initial_training: ', stop_initial_train - start_initial_train)  

    

  @staticmethod
  def collapse(inputs, rho_res):
    rho_h = jnp.matmul(jnp.conj(inputs), rho_res)
    rho_res = jnp.einsum(
        '...i, ...i -> ...',
        rho_h, jnp.conj(rho_h), 
        optimize='optimal') # shape (b,)
    #rho_res = jnp.dot(rho_h, jnp.conj(rho_h))
    return rho_res

  @staticmethod
  def obtain_params_batches(values, batch_size, key):
    num_train = values.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

  

  @partial(jit, static_argnums=(0,))
  def predict(self, values):
    num_batches = InqMeasurement.obtain_params_batches(values, self.batch_size, self.key)
    results = None
    rho_res = self.rho_res / self.num_samples
    num_train = values.shape[0]
    perm = jnp.arange(num_train)
    for i in range(num_batches):
      batch_idx = perm[i * self.batch_size: (i + 1)*self.batch_size]
      batch = values[batch_idx, :]

      inputs = self.fm_x(batch)
      batch_probs = self.collapse_batch(inputs, rho_res)
      results = jnp.concatenate([results, batch_probs], axis=0) if results is not None else batch_probs
    return results

