import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common
import common.transformer
import common.transformer_float16


class EnsembleRSSM(common.Module):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='elu', norm='none', obs_out_norm='none', std_act='softplus', min_std=0.1, ar_steps=0):
    super().__init__()
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._obs_out_norm = obs_out_norm
    self._std_act = std_act
    self._min_std = min_std
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)
    self.ar_steps = ar_steps
    self.transformer = common.transformer.Transformer(num_layers=1,
                                                      d_model=self._discrete,
                                                      num_heads=1,
                                                      dff=self._discrete*self._stoch,
                                                      input_vocab_size=1000,
                                                      target_vocab_size=1000,
                                                      pe_input=self._deter,
                                                      pe_target=self._discrete,
                                                      rate=0)
    self.transformer_decoder = common.transformer.Decoder(num_layers=1, d_model=self._discrete, num_heads=1,
                         dff=self._discrete*self._stoch, target_vocab_size=320,
                         maximum_position_encoding=self._discrete)
    self.transformer_decoder_float16 = common.transformer_float16.Decoder(num_layers=1, d_model=self._discrete, num_heads=1,
                         dff=self._discrete*self._stoch, target_vocab_size=320,
                         maximum_position_encoding=self._discrete)
    
    
    # num_layers = 4
    # d_model = 128
    # dff = 512
    # num_heads = 8
    # dropout_rate = 0
    
    # self.transformer = Transformer(
    # num_layers=num_layers,
    # d_model=d_model,
    # num_heads=num_heads,
    # dff=dff,
    # input_vocab_size=self._discrete,
    # target_vocab_size=self._discrete,
    # pe_input=self._stoch,
    # pe_target=self._stoch,
    # rate=dropout_rate)
    # self.autoreg = AutoRegressiveDecoderLayer(d_model=deter, num_heads=8, dff=2048, dropout_rate=0, loops = 5, prior_dim=stoch)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape)))) # Swapping index B (batch) with index L (length) - example: [B, L, W, H, C] -> [L, B, W, H, C]
    if state is None:
      state = self.initial(tf.shape(action)[0])
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (swap(action), swap(embed), swap(is_first)), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior
  
  # @tf.function
  # def autoreg_observe(self, embed, action, is_first, prev_state=None, prev_prior=None, ar_steps=0):
    
  #   if prev_state is None:
  #     prev_state = self.initial(tf.shape(action)[0])
      
  #   shape_state = prev_state['stoch'].shape[:-2] + [self._stoch * self._discrete]
  #   shape_state = prev_state['stoch'].shape[:-2] + [self._stoch * self._discrete]
  #   swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    
  #   for _ in range(ar_steps):
    
  #     ################################################# Function(state_i, prior_i) -> state_i+1 #################################################
      
  #     for k, v_p in prev_prior.items:
  #       prev_prior[k] = swap(prev_prior[k])
        
  #     for step_i in range(len(prev_prior['deter'])):
  #       for ((k, v_s), (_, v_p)) in zip(prev_state.items(), prev_prior.items()):
  #         if k == "deter":
  #           new_state[k] = self.get(f'autoreg_state_{k}', tfkl.Dense, v_s.shape[-1])(v_s) + self.get(f'autoreg_prior_{k}', tfkl.Dense, v_s.shape[-1])(v_p)
  #         else:
  #           new_state[k] = self.get(f'autoreg_state_{k}', tfkl.Dense, shape[-1])(tf.reshape(v_s, shape)) + self.get(f'autoreg_prior_{k}', tfkl.Dense, shape[-1])(tf.reshape(v_p, shape))
  #           new_state[k] = tf.reshape(new_state[k], v_s.shape)
  #       ###########################################################################################################################################
        
        
  #       ################################################# Function(state_i+1) -> prior_i+1 #########################################################
  #       swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape)))) # Swapping index B (batch) with index L (length) - example: [B, L, W, H, C] -> [L, B, W, H, C]
  #       if new_state is None:
  #         new_state = self.initial(tf.shape(action)[0])
  #       prior = common.static_scan(
  #           lambda prev, inputs: self.autoreg_obs_step(prev[0], *inputs),
  #           (swap(action), swap(embed), swap(is_first)), (new_state, new_state))
  #       prior = {k: swap(v) for k, v in prior.items()}
        
  #       prev_state = new_state
  #       prev_prior = prior
  #     ###########################################################################################################################################
  #   return prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    # self.autoreg(prior, state)
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
    # if is_first.any():
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum(
            'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        (prev_state, prev_action))
    prior = self.img_step(prev_state, prev_action, sample)

    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._obs_out_norm)(x)
    x = self._act(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    
    # Autoregressive Prior:
    # for _ in range(self.ar_steps):
    #   # z to h
    #   x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    #   x = self.get('obs_out_norm', NormLayer, self._obs_out_norm)(x)
    #   x = self._act(x)
    
    return post, prior
  
  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get(f'img_in', tfkl.Dense, self._hidden)(x)
    x = self.get(f'img_in_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter]) # GRUCell takes an input as (inputs, states) and returns (h, new_state)
    deter = deter[0]  # Keras wraps the state in a list
    
    # Auto-Regressive Prior Part
    # h = [self.autoregGRU.get_initial_state(None, x.shape[0], x.dtype)]
    # for _ in range(self.ar_steps):
    #   x, h = self.autoregGRU(x, h)
    
    stats = self._suff_stats_ensemble(x)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    
    # Reshape logits to feed to Dense laeyrs:
    if self._discrete and self.ar_steps > 0:
      # h = deter
      # z = stats['logit']
      # shape = stats['logit'].shape[:-2] + [self._stoch * self._discrete]
      # z = tf.reshape(stats['logit'], shape)
      # z = stats['logit']
      # enc_padding_mask, combined_mask, dec_padding_mask = common.transformer.create_masks(h[:-1], stats['logit'][:-1])
      transDecoder = self.transformer_decoder if deter.dtype == tf.float32 else self.transformer_decoder_float16
      for _ in range(self.ar_steps):
        d = self.get(f'transformer_in', tfkl.Dense, self._discrete*3)(deter)
        stats['logit'], _ = transDecoder(stats['logit'], tf.reshape(d, [d.shape[0], 3, self._discrete]),
                            True,
                            None,
                            None)
      # stats['logit'], _ = self.transformer(deter, stats['logit'],
      #                     True,
      #                     None,
      #                     None,
      #                     None)
      
      # for _ in range(self.ar_steps): 
      #   # h = self.get("autoreg_logit2deter", tfkl.Dense, self._deter, None)(z) + self.get('autoreg_deter2deter', tfkl.Dense, self._deter)(h)
      #   h = self.get("autoreg_concat_logit_deter", tfkl.Dense, self._deter, None)(tf.concat([z, h, prev_action], -1))
      #   h = self.get('autoreg_deter_norm', NormLayer, self._norm)(h)
      #   h = self._act(h)
        
      #   z = self.get("autoreg_deter2logit", tfkl.Dense, self._stoch * self._discrete, None)(h)
      #   z = self.get('autoreg_logit_norm', NormLayer, self._norm)(z)
      #   z = self._act(z)
      #   z = self.get('autoreg_deter2logit2', tfkl.Dense, self._stoch * self._discrete, None)(z)
      
      # Reshape logits back
      # stats['logit'] = tf.reshape(z, z.shape[:-1] + [self._stoch, self._discrete])  
      dist = self.get_dist(stats)
      stoch = dist.sample() if sample else dist.mode()
    else:
      dist = self.get_dist(stats)
      stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior
    
    # Auto-Regressive Prior Part
    # h = [self.autoregGRU.get_initial_state(None, x.shape[0], x.dtype)]
    # for _ in range(self.ar_steps):
    #   x, h = self.autoregGRU(x, h)    
    

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class Encoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    self.shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Encoder CNN inputs:', list(self.cnn_keys))
    print('Encoder MLP inputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

  @tf.function
  def __call__(self, data):
    key, shape = list(self.shapes.items())[0]
    batch_dims = data[key].shape[:-len(shape)]
    data = {
        k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_keys:
      outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
    if self.mlp_keys:
      outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
    output = tf.concat(outputs, -1)
    return output.reshape(batch_dims + output.shape[1:])

  def _cnn(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** i * self._cnn_depth
      x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x.reshape(tuple(x.shape[:-3]) + (-1,))

  def _mlp(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}', tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x

# class AutoRegressive(common.Module):

#   def __init__(
#       self, prior, h, mlp_keys=r'.*', act='elu', norm='none',
#       mlp_layers=[400, 400, 400, 400]):
#     self.shapes = shapes
#     self.cnn_keys = [
#         k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
#     self.mlp_keys = [
#         k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
#     print('Encoder CNN inputs:', list(self.cnn_keys))
#     print('Encoder MLP inputs:', list(self.mlp_keys))
#     self._act = get_act(act)
#     self._norm = norm
#     self._cnn_depth = cnn_depth
#     self._cnn_kernels = cnn_kernels
#     self._mlp_layers = mlp_layers

#   @tf.function
#   def __call__(self, data):
#     key, shape = list(self.shapes.items())[0]
#     batch_dims = data[key].shape[:-len(shape)]
#     data = {
#         k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
#         for k, v in data.items()}
#     outputs = []
#     if self.cnn_keys:
#       outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
#     if self.mlp_keys:
#       outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
#     output = tf.concat(outputs, -1)
#     return output.reshape(batch_dims + output.shape[1:])

#   def _cnn(self, data):
#     x = tf.concat(list(data.values()), -1)
#     x = x.astype(prec.global_policy().compute_dtype)
#     for i, kernel in enumerate(self._cnn_kernels):
#       depth = 2 ** i * self._cnn_depth
#       x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
#       x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
#       x = self._act(x)
#     return x.reshape(tuple(x.shape[:-3]) + (-1,))

#   def _mlp(self, data):
#     x = tf.concat(list(data.values()), -1)
#     x = x.astype(prec.global_policy().compute_dtype)
#     for i, width in enumerate(self._mlp_layers):
#       x = self.get(f'dense{i}', tfkl.Dense, width)(x)
#       x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
#       x = self._act(x)
#     return x


class Decoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    self._shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Decoder CNN outputs:', list(self.cnn_keys))
    print('Decoder MLP outputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

  def __call__(self, features):
    features = tf.cast(features, prec.global_policy().compute_dtype)
    outputs = {}
    if self.cnn_keys:
      outputs.update(self._cnn(features))
    if self.mlp_keys:
      outputs.update(self._mlp(features))
    return outputs

  def _cnn(self, features):
    channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
    ConvT = tfkl.Conv2DTranspose
    x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
      act, norm = self._act, self._norm
      if i == len(self._cnn_kernels) - 1:
        depth, act, norm = sum(channels.values()), tf.identity, 'none'
      x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}', NormLayer, norm)(x)
      x = act(x)
    x = x.reshape(features.shape[:-1] + x.shape[1:])
    means = tf.split(x, list(channels.values()), -1)
    dists = {
        key: tfd.Independent(tfd.Normal(mean, 1), 3)
        for (key, shape), mean in zip(channels.items(), means)}
    return dists

  def _mlp(self, features):
    shapes = {k: self._shapes[k] for k in self.mlp_keys}
    x = features
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}', tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    dists = {}
    for key, shape in shapes.items():
      dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
    return dists


class MLP(common.Module):

  def __init__(self, shape, layers, units, act='elu', norm='none', **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = get_act(act)
    self._out = out

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
      x = self.get(f'norm{index}', NormLayer, self._norm)(x)
      x = self._act(x)
    x = x.reshape(features.shape[:-1] + [x.shape[-1]])
    return self.get('out', DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(common.Module):

  def __init__(
      self, shape, dist='mse', min_std=0.1, init_std=0.0):
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

  def __call__(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    raise NotImplementedError(self._dist)


class NormLayer(common.Module):

  def __init__(self, name):
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      self._layer = tfkl.LayerNormalization()
    elif name == 'batchnorm':
      self._layer = tfkl.BatchNormalization()
    else:
      raise NotImplementedError(name)

  def __call__(self, features):
    if not self._layer:
      return features
    return self._layer(features)


def get_act(name):
  if name == 'none':
    return tf.identity
  if name == 'mish':
    return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  elif hasattr(tf, name):
    return getattr(tf, name)
  else:
    raise NotImplementedError(name)

# Auto-Regressive Part

# class BaseAttention(tf.keras.layers.Layer):
#   def __init__(self, **kwargs):
#     super().__init__()
#     self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
#     self.layernorm = tf.keras.layers.LayerNormalization()
#     self.add = tf.keras.layers.Add()

# class CausalSelfAttention(BaseAttention):
#   def call(self, x):
#     attn_output = self.mha(
#         query=x,
#         value=x,
#         key=x
#         )
#     x = self.add([x, attn_output])
#     x = self.layernorm(x)
#     return x
  
# class CrossAttention(BaseAttention):
#   def call(self, x, context):
#     attn_output, attn_scores = self.mha(
#         query=x,
#         key=context,
#         value=context,
#         return_attention_scores=True)

#     # Cache the attention scores for plotting later.
#     self.last_attn_scores = attn_scores

#     x = self.add([x, attn_output])
#     x = self.layernorm(x)

#     return x

# class FeedForward(tf.keras.layers.Layer):
#   def __init__(self, d_model, dff, dropout_rate=0.1):
#     super().__init__()
#     self.seq = tf.keras.Sequential([
#       tf.keras.layers.Dense(dff, activation='relu'),
#       tf.keras.layers.Dense(d_model),
#       tf.keras.layers.Dropout(dropout_rate)
#     ])
#     self.add = tf.keras.layers.Add()
#     self.layer_norm = tf.keras.layers.LayerNormalization()

#   def call(self, x):
#     x = self.add([x, self.seq(x)])
#     x = self.layer_norm(x) 
#     return x

# class AutoRegressiveDecoderLayer(tf.keras.layers.Layer):
#   def __init__(self,
#                *,
#                d_model,
#                num_heads,
#                dff,
#                dropout_rate=0.1,
#                loops = 5,
#                prior_dim=30):
#     super(AutoRegressiveDecoderLayer, self).__init__()

#     self.causal_self_attention = CausalSelfAttention(
#         num_heads=num_heads,
#         key_dim=d_model,
#         dropout=dropout_rate)

#     self.cross_attention = CrossAttention(
#         num_heads=num_heads,
#         key_dim=d_model,
#         dropout=dropout_rate)

#     self.ffn = FeedForward(d_model, dff)
    
#     self.prior2h = tf.keras.layers.Dense(d_model)
#     self.h2prior = tf.keras.layers.Dense(prior_dim)
#     self.flatten = tfkl.Flatten()
    
#     self.loops = loops

#   def call(self, prior, state):
#     h_prev = state['deter']
    
#     for _ in range(self.loops):
#       prior_flattened = self.flatten(prior['stoch'])
#       h = self.prior2h(prior_flattened)
#       h = self.causal_self_attention(x=h)
#       print(f'{"PRINTED TEXT":.^100}')
#       print(h.shape)
#       print(h_prev.shape)
#       print(f'{"END PRINTED TEXT":.^100}')
#       h = self.cross_attention(x=h, context=h_prev)

#       h_prev = self.ffn(h)
#       prior['stoch'] = self.h2prior(h_prev)
#     # return prior