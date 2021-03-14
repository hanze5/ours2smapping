from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self):
        # 这里的命名可以区分variable归属于哪一层
        name = self.__class__.__name__.lower()
        self.name = name
        self.vars = {}
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, sparse_inputs=False, act=tf.nn.relu,
                 bias=False, featureless=False):
        super(GraphConvolution, self).__init__()
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # 这里variable_scope都是基于外层model中的variable_scope
        # glorot函数返回一个用Glorot方法初始化的Variable
        # zeros函数返回一个均为0的Variable
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_0'] = glorot([input_dim, output_dim], name='weights_0')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs
        # self.support是邻接矩阵
        # 如果self.support里有2个邻接矩阵，那么这个图卷积层会迭代两轮邻居的数据，依次储存在
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_0'], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_0']
        output = dot(self.support, pre_sup, sparse=True)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output#output
        return self.act(output)
