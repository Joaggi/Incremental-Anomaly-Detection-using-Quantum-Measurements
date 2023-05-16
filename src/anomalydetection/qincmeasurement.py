from tensorflow import keras
import qmc.tf.layers as layers
import qmc.tf.models as models
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

class QMeasureDensity(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input  state
    """

    def __init__(
            self,
            dim_x: int,
            **kwargs
    ):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.rho = self.add_weight(
            "rho",
            shape=(self.dim_x, self.dim_x),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        oper = tf.einsum(
            '...i,...j->...ij',
            inputs, tf.math.conj(inputs),
            optimize='optimal') # shape (b, nx, nx)
        rho_res = tf.einsum(
            '...ik, km, ...mi -> ...',
            oper, self.rho, oper,
            optimize='optimal')  # shape (b, nx, ny, nx, ny)
        return rho_res

    @tf.function
    def call_train(self, x):
        if not self.qmd.built:
            self.call(x)
        psi = self.fm_x(x)
        rho = self.cp([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def compute_output_shape(self, input_shape):
        return (1,)

class QIncMeasurement(tf.keras.Model):
    """
    An Incremental Anomaly Detection through Quantum Measurement.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
    """
    def __init__(self, fm_x, dim_x, remember_rate=0.99):
        super(QIncMeasurement, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.r_rate =  remember_rate
        self.qmd = QMeasureDensity(dim_x)
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        return probs

    @tf.function(jit_compile=True)
    def call_train(self, x):
        if not self.qmd.built:
            self.call(x)
        psi = self.fm_x(x)
        rho = self.cp([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        data =  data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        if x.shape[1] is not None:
            rho = self.call_train(x)
            self.qmd.weights[0].assign(self.qmd.weights[0] + rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QIncMeasurement, self).fit(*args, **kwargs)
        self.qmd.weights[0].assign(self.qmd.weights[0] / self.num_samples)
        return result

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

