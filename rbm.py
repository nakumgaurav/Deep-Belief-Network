from __future__ import print_function

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy as np
import scipy.misc
import tensorflow as tf
import os


class RBM(object):
	def __init__(self, 
		input=None, 
		n_visible=784, 
		n_hidden=500, 
		W=None, 
		hbias=None, 
		vbias=None,
		numpy_rng=None,
		tf_rng=None,
	):
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1234)

		if tf_rng is None:
			tf_rng = 41

		if W is None:
			# print("HUHIHHAAAAAAAAAAAAHUHIHHAAAAAAAAAAAAHUHIHHAAAAAAAAAAAAHUHIHHAAAAAA")
			initial_W = np.asarray(numpy_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)), high=4 * np.sqrt(6. / (n_hidden + n_visible)), 
			size=(n_visible, n_hidden)))

			# if(n_visible==784):
			# 	np.save("initial_W.npy", initial_W)

			W = tf.get_variable("W", dtype=tf.float32, initializer=tf.cast(tf.constant(initial_W), dtype=tf.float32), trainable=True)

		if hbias is None:
			hbias = tf.get_variable("hbias", dtype=tf.float32, initializer=tf.constant(np.zeros(n_hidden, dtype=np.float32)), trainable=True)

		if vbias is None:
			vbias = tf.get_variable("vbias", dtype=tf.float32, initializer=tf.constant(np.zeros(n_visible, dtype=np.float32)), trainable=True)

		if input is None:
			self.input = tf.placeholder(dtype=tf.float32, shape=[None, n_visible], name="input")
		else:
			self.input = input

		self.W = W
		self.hbias = hbias
		self.vbias = vbias

		self.numpy_rng = numpy_rng
		self.tf_rng = tf_rng

       	# self.params = [self.W, self.hbias, self.vbias]

	def propup(self, vis):
		with tf.variable_scope("propup"):
			# print("W shape:", self.W.shape, "vis shape:", vis.shape)
			pre_sigmoid_activation = tf.matmul(vis, self.W) + self.hbias
			return tf.nn.sigmoid(pre_sigmoid_activation)

	def propdown(self, hid):
		with tf.variable_scope("propdown"):
			pre_sigmoid_activation = tf.matmul(hid, tf.transpose(self.W)) + self.vbias
			return tf.nn.sigmoid(pre_sigmoid_activation)

	def sample_h_given_v(self, v_sample):
		with tf.variable_scope("sample_h_given_v"):
			h_prob = self.propup(v_sample)
			# print("h_prob.shape=", h_prob.shape)
			berno_obj = tf.distributions.Bernoulli(probs=h_prob, dtype=tf.float32)
			h_sample = berno_obj.sample(seed=self.tf_rng, name='h_sample')

			return h_sample

	def sample_v_given_h(self, h_sample):
		with tf.variable_scope("sample_v_given_h"):
			v_prob = self.propdown(h_sample)
			berno_obj = tf.distributions.Bernoulli(probs=v_prob, dtype=tf.float32)
			v_sample = berno_obj.sample(seed=self.tf_rng, name='v_sample')

			return v_sample

	def gibbs_hvh(self, h0_sample):
		''' This function performs one step of Gibbs sampling
			starting from the hidden state'''

		v1_sample = self.sample_v_given_h(h0_sample)
		h1_sample = self.sample_h_given_v(v1_sample)

		return (v1_sample, h1_sample)

	def gibbs_vhv(self, v0_sample):
		''' This function performs one step of Gibbs sampling
			starting from the visible state'''

		h1_sample = sample_h_given_v(v0_sample)
		v1_sample = sample_v_given_h(h1_sample)

		return (h1_sample, v1_sample)

	def free_energy(self, v_sample):
		with tf.variable_scope("free_energy"):
			wx_b = tf.matmul(v_sample, self.W) + self.hbias
			return -tf.tensordot(v_sample, self.vbias, axes=1) -tf.reduce_sum(tf.log(1 + tf.exp(wx_b)), axis=1)

	def get_grads(self, k=5):
		# positive phase
		ph_sample = self.sample_h_given_v(self.input)
		chain_start = ph_sample

		temp_h_sample = tf.identity(chain_start)
		for i in range(k-1):
			_, temp_h_sample = self.gibbs_hvh(temp_h_sample)
		nv_sample, nh_sample = self.gibbs_hvh(temp_h_sample)

		# chain_end = tf.get_variable(name="chain_end", dtype=tf.float32, initializer=tf.constant(nv_sample), trainable=False)
		chain_end = nv_sample
		cost = tf.reduce_mean(self.free_energy(self.input) - self.free_energy(chain_end))

		h_props = self.propup(chain_end)
		w_positive_grad = tf.matmul(tf.transpose(self.input), chain_start)
		w_negative_grad = tf.matmul(tf.transpose(chain_end), h_props)
		w_grad = (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(self.input)[0])
		vb_grad = tf.reduce_mean(self.input - chain_end, 0)
		hb_grad = tf.reduce_mean(chain_start - h_props, 0)

		# h0_props = self.propup(self.input)
		# w_positive_grad = tf.matmul(tf.transpose(self.input), h0_props)
		# w_negative_grad = tf.matmul(tf.transpose(nv_sample), nh_sample)
		# w_grad = (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(self.input)[0])
		# vb_grad = tf.reduce_mean(self.input - nv_sample, 0)
		# hb_grad = tf.reduce_mean(h0_props - nh_sample, 0)

		return w_grad, vb_grad, hb_grad

	def update_params(self, lr=0.1):
		w_grad, vb_grad, hb_grad = self.get_grads()

		# updates the weights and biases
		update_w = tf.assign(self.W, self.W + lr * w_grad)
		update_vb = tf.assign(self.vbias, self.vbias + lr * vb_grad)
		update_hb = tf.assign(self.hbias, self.hbias + lr * hb_grad)

		return update_w, update_vb, update_hb

	def sampler(self, input=None, steps=5000):
		if input is None:
			v_samples = self.input
		else:
			v_samples = input
		for step in range(steps):
			v_samples = self.sample_v_given_h(self.sample_h_given_v(v_samples))
		return v_samples


def save_images(images, size, path):
	# img = (images + 1.0) / 2.0
	img = images
	h, w = img.shape[1]*2, img.shape[2]*2
	
	merge_img = np.zeros((h * size[0], w * size[1]))
	
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		merge_img[j*h:j*h+h, i*w:i*w+w] = scipy.misc.imresize(image, size=2.0, interp='bicubic')
	
	return scipy.misc.imsave(path, merge_img)

		# i = idx % size[1]
		# i = (i*4) % size[1]
		# j = (idx*4) // size[1]
		# merge_img[j*h:j*h+4*h, i*w:i*w+4*w] = scipy.misc.imresize(image, size=4.0, interp='bicubic')

# def train():
# 	log_dir = './logs'
# 	samples_dir = './samples'

# 	epochs = 50
# 	batch_size = 64

# 	from tensorflow.examples.tutorials.mnist import input_data
# 	mnist = input_data.read_data_sets("MNIST_data/")

# 	# mnist = tf.keras.datasets.mnist
# 	# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 	# x_train, x_test = x_train/255.0, x_test/255.0

# 	x_train = mnist.train
# 	num_train = x_train.num_examples
# 	# print("num_train=",num_train)
# 	num_batches = num_train // batch_size
# 	x = tf.placeholder(tf.float32, shape=[None, 784], name="input")

# 	rbm = RBM(x)
# 	step = rbm.update_params()
# 	print("Sampling images...")
# 	sampler = rbm.sampler(x)

# 	# saver = tf.train.Saver()


# 	with tf.Session() as sess:
# 		print("Initializing variables...")
# 		init = tf.global_variables_initializer()
# 		sess.run(init)
# 		for i in range(epochs * num_batches):
# 			batch_x, _ = x_train.next_batch(batch_size)
# 			batch_x[batch_x<0.5] = 0.0
# 			batch_x[batch_x>=0.5] = 1.0
# 			# my_value = batch_x[0, 100:400]
# 			# print("my_value:", my_value)
# 			# sess.run(my_value)
# 			# draw samples
# 			if i % 500 == 0:
# 				print("Iteration %d" %i)
# 				samples = sess.run(sampler, feed_dict = {x: batch_x})
# 				# samples = batch_x
# 				samples = samples.reshape([batch_size, 28, 28])
# 				save_images(samples, [8, 8], os.path.join(samples_dir, 'iteration_%d.png' % i))

# 			sess.run(step, feed_dict = {x:batch_x})

# 		# draw samples when training finished
# 		print('Test')
# 		samples = sess.run(sampler, feed_dict = {x: batch_x})
# 		samples = samples.reshape([batch_size, 28, 28])
# 		save_images(samples, [8, 8], os.path.join(samples_dir, 'test.png'))
# 		print('Saved samples.')

# train()