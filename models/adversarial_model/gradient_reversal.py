import tensorflow as tf


class GradientReversalLayer(tf.keras.layers.Layer):
    """The gradient reversal layer is a layer that multiplies the gradient by a negative constant during
    backpropagation.
    Args:
      lambda_: Float32, the constant by which the gradient is multiplied. It should be a negative number.
    References:
      https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0
      https://www.tensorflow.org/guide/eager#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A2%AF%E5%BA%A6
    """
    def __init__(self, total_calls):
        super().__init__(trainable=False, name="gradient_reversal_layer")
        self.lambda_ = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.total_calls = tf.constant(total_calls, dtype=tf.float32)

    def call(self, x, **kwargs):
        return self.grad_reversed(x)

    def build(self, input_shape):
        self.calls = tf.Variable(0, dtype=tf.float32, trainable=False)

    @tf.custom_gradient
    def grad_reversed(self, x):
        """
        It returns input and a custom gradient function.
        Args:
          x: The input tensor.
        Returns:
          the input x and the custom gradient function.
        """
        y = tf.identity(x)

        def custom_gradient(dy):
            self.calls.assign_add(1)
            self.update_lambda()
            return tf.negative(self.lambda_ * dy)

        return y, custom_gradient

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "lambda": float(self.lambda_.numpy())
        })
        return config

    def get_calls(self):
        return self.calls

    def update_lambda(self):

        self.lambda_ = tf.subtract(tf.divide(tf.constant(2.),
                                             tf.add(tf.constant(1.),
                                                    tf.math.exp(tf.negative(tf.multiply(tf.constant(10.),
                                                                                        tf.divide(self.calls,
                                                                                                  self.total_calls)
                                                                                        )
                                                                            )
                                                                )
                                                    )
                                             ),
                                   tf.constant(1.)
                                   )
