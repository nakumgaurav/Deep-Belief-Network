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
from rbm import RBM, save_images
from dataset_util import Dataset

log_dir = './logs'
samples_dir = './samples'

class DBN(object):
	def __init__(self,
		inputs,
		n_ins = 9,
		hidden_layers_sizes = [100,100],
		n_outs = 3,
		rbm_epochs=2,
		rbm_batch_size=10,
		dbn_epochs=30,
		dbn_batch_size=10,
		learning_rate=0.001,
		numpy_rng=None,
		tf_rng=None
	):
		self.n_visible = n_ins
		self.hidden_layers_sizes = hidden_layers_sizes
		self.n_output = n_outs
		self.l = len(hidden_layers_sizes)

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1234)
		self.numpy_rng = numpy_rng

		if tf_rng is None:
			tf_rng = 41
		self.tf_rng = tf_rng

		self.inputs = inputs

		self.rbm_epochs = rbm_epochs
		self.rbm_batch_size = rbm_batch_size
		self.dbn_epochs = dbn_epochs
		self.dbn_batch_size = dbn_batch_size

		self.learning_rate = learning_rate

		self.pre_trained_DBN = {}

		self.visible_units_list = [n_ins]
		for i in range(len(hidden_layers_sizes)-1):
			self.visible_units_list.append(hidden_layers_sizes[i])

		self.rbm_stack = []

		self.gen_weights = []
		self.gen_biases = []
		self.rec_weights = []
		self.rec_biases = []


	# a bottom-up pass through the DBN to get the top level hidden probabilties
	def propup_dbn(self, inputs):
		def sample_h_given_v(v_sample, W, hbias):
			h_prob = tf.nn.sigmoid(tf.matmul(v_sample, W) + hbias)
			# print("h_prob.shape=", h_prob.shape)
			berno_obj = tf.distributions.Bernoulli(probs=h_prob, dtype=tf.float32)
			h_sample = berno_obj.sample(seed=self.tf_rng, name='h_sample')

			return h_sample, h_prob

		v = tf.identity(inputs)
		h_prob = None

		v_samp_layers = [v]
		h_prob_layers = [h_prob]
		# propup
		for i, W in enumerate(self.rec_weights):
			with tf.variable_scope("propup_dbn%d"%i):
				v, h_prob = sample_h_given_v(v, W, self.rec_biases[i])
				v_samp_layers.append(v)
				h_prob_layers.append(h_prob)

		return v_samp_layers, h_prob_layers


	# an up-down pass through the DBN to get the reconstruction probabilties
	def propdown_dbn(self, inputs):
		def sample_v_given_h(h_sample, W, vbias):
			v_prob = tf.nn.sigmoid(tf.matmul(h_sample, W) + vbias)
			berno_obj = tf.distributions.Bernoulli(probs=v_prob, dtype=tf.float32)
			v_sample = berno_obj.sample(seed=self.tf_rng, name='v_sample')

			return v_sample, v_prob

		h = tf.identity(inputs)
		v_prob = None

		h_samp_layers = [h]
		v_prob_layers = [v_prob]
		# propup
		for i, W in enumerate(self.gen_weights):
			with tf.variable_scope("propdown_dbn%d"%i):
				h, v_prob = sample_v_given_h(h, W, self.gen_biases[i])
				h_samp_layers.append(h)
				v_prob_layers.append(v_prob)

		return h_samp_layers, v_prob_layers


	def update_gen_params(self, gen_samps, gen_probs, rec_samps, rec_probs):


	def update_rec_params(self, gen_samps, gen_probs, rec_samps, rec_probs):


	def wake_sleep(self, inputs):
		rec_samps, rec_probs = self.propup_dbn(inputs)
		gen_samps, gen_probs = self.propdown_dbn(rec_samps[-1])

		step_gen = self.update_gen_params(gen_samps, gen_probs, rec_samps, rec_probs)
		step_rec = self.update_rec_params(gen_samps, gen_probs, rec_samps, rec_probs)

		return step_gen, step_rec


	# Given the inputs, propagate them through the DBN and
	# get back the reconstructed output
	def prop_dbn(self, inputs):
		v = tf.identity(inputs)
		# propup
		for i in range(self.l):
			rbm = self.rbm_stack[i]
			v = rbm.sample_h_given_v(v)

		h = tf.identity(v)
		# propdown
		for i in range(self.l):
			rbm = self.rbm_stack[self.l-i-1]
			if(i<self.l-1):
				h = rbm.sample_v_given_h(h)
			else:
				h = rbm.propdown(h)

		return h

	def sampler(self, inputs=None, num_steps=500):
		if inputs is None:
			v_samples = tf.identity(self.inputs)
		else:
			v_samples = tf.identity(inputs)
		for step_no in range(num_steps):
			v_samples = self.prop_dbn(v_samples)
		
		return v_samples

	def construct_rbm(self, x):
			steps = []
			for i in range(self.l):
				with tf.variable_scope("RBM%d"%(i+1)) as scope:
					rbm = RBM(x, n_visible=self.visible_units_list[i], n_hidden=self.hidden_layers_sizes[i], 
						numpy_rng=self.numpy_rng, tf_rng=self.tf_rng)
					scope.reuse_variables()
					self.rbm_stack.append(rbm)

					step = rbm.update_params()
					steps.append(step)

			return steps





	def train(self):
		x = tf.placeholder(tf.float32, shape=[None, None], name="input")

		num_train = self.inputs._num_examples
		num_rbm_batches = num_train // self.rbm_batch_size
		num_dbn_batches = num_train // self.dbn_batch_size

		steps = self.construct_rbm(x)

		def untie_weights():
			for i, rbm in enumerate(self.rbm_stack):
				w_gen = tf.get_variable("w_gen%d"%i, dtype=tf.float32, initializer=tf.transpose(rbm.W), trainable=True)
				w_rec = tf.get_variable("w_rec%d"%i, dtype=tf.float32, initializer=rbm.W, trainable=True)
				bias_gen = tf.get_variable("bias_gen_%d"%i, dtype=tf.float32, initializer=rbm.vbias, trainable=True)
				bias_rec = tf.get_variable("bias_rec_%d"%(i+1), dtype=tf.float32, initializer=rbm.hbias, trainable=True)
				self.gen_weights.append(w_gen)
				self.rec_weights.append(w_rec)
				self.gen_biases.append(bias_gen)
				self.rec_biases.append(bias_rec)


		# propagate the input up to the RBM which is to be trained now, 
		# using the k RBMs below which are already trained 
		def transform_batch(k):
			temp = x
			for layer_no in range(k):
				rbm = self.rbm_stack[layer_no]
				temp = rbm.sample_h_given_v(temp)
			return temp

		with tf.Session() as sess:
			# Pre-training
			print("Initializing variables...")
			init = tf.global_variables_initializer()
			sess.run(init)

			print("Greedy Layerwise Unsupervised Pre-training...")

			for i in range(self.l):
				print("RBM %d"%(i+1))
				z = transform_batch(i)
				with tf.variable_scope("RBM%d"%(i+1), reuse=True):
					for epoch_no in range(self.rbm_epochs):
						print("Epoch ", epoch_no+1)
						for batch_no in range(num_rbm_batches):
							# print("Batch ", batch_no+1)
							batch_x = self.inputs.get_next_batch(self.rbm_batch_size)
							batch_x = sess.run(z, feed_dict={x: batch_x})
							# batch_x[batch_x<0.5] = 0.0
							# batch_x[batch_x>=0.5] = 1.0
							sess.run(steps[i], feed_dict={x: batch_x})

				for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RBM%d"%(i+1)):
					self.pre_trained_DBN[var.name] = sess.run(var)

				self.rbm_stack[i].W = self.pre_trained_DBN["RBM%d/W:0"%(i+1)]
				self.rbm_stack[i].hbias = self.pre_trained_DBN["RBM%d/hbias:0"%(i+1)]
				self.rbm_stack[i].vbias = self.pre_trained_DBN["RBM%d/vbias:0"%(i+1)]


			### Check if learning is occuring
			# w1 = self.pre_trained_DBN["RBM1/W:0"]
			# # print(w2)
			# w1_ini = np.load("initial_W.npy")
			# # print(w2_ini)
			# # print(w2_ini)
			# print(np.linalg.norm(w1-w1_ini))

			untie_weights()

			print("Setting up sampler...")
			sampler = self.sampler(x)

			step = wake_sleep(x)

			print("Fine-tuning...")
			for epoch_no in range(self.dbn_epochs):
				print("Epoch # ", epoch_no+1)
				for batch_no in range(num_dbn_batches):
					batch_x = self.inputs.get_next_batch(self.dbn_batch_size)
					batch_x[batch_x<0.5] = 0.0
					batch_x[batch_x>=0.5] = 1.0

					sess.run(step, feed_dict = {x:batch_x})

				# # draw samples
				# samples = sess.run(sampler, feed_dict = {x: batch_x})
				# # samples = batch_x
				# samples = samples.reshape([batch_size, 28, 28])
				# save_images(samples, [8, 8], os.path.join(samples_dir, 'Epoch%d.png' % epoch_no))


			# draw samples when training finished
			# batch_x = self.inputs.get_next_batch(self.dbn_batch_size)
			# print('Test')
			# samples = sess.run(sampler, feed_dict = {x: batch_x})
			# samples = samples.reshape([self.dbn_batch_size, 28, 28])
			# batch_x = batch_x.reshape([self.dbn_batch_size, 28, 28])
			# save_images(samples, [4, 4], os.path.join(samples_dir, 'recon.png'))
			# save_images(batch_x, [4, 4], os.path.join(samples_dir, 'orig.png'))
			# print('Saved samples.')






# def load_dataset():
# 	pass

def main():
	# MNIST
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/")
	X_train = mnist.train
	train_data = X_train._images
	dataset = Dataset(train_data[:10000])	
	# dataset = load_dataset()
	# X_train, y_train, X_test, y_test = dataset
	# layers = [100,100]

	dbn = DBN(inputs=dataset, n_ins=784, hidden_layers_sizes=[500, 500], n_outs=10, rbm_batch_size=64, dbn_batch_size=16)
	dbn.train()

if __name__ == "__main__":
	main()

