import operator
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Layer


class OctantClasses():
    '''
	Convenient mapping between vector of 3 signs
	to corresponding classes in sparse representation.
	e.g. (-1, -1, -1) maps to class 0.
	Encodes 8 classes (8 possible octants)
    '''
    def __init__(self):
        classes_dict_keys = list(range(8))
        classes_dict_items = []

        vals = [-1, 1]
        for i in vals:
            for j in vals:
                for k in vals:
                   classes_dict_items.append((i,j,k))

        self.classes_dict = dict(zip(classes_dict_keys, classes_dict_items))
        self.classes_dict_inv = dict(zip(classes_dict_items, classes_dict_keys))

    def signs_to_classes(self, signs):
		'''
		Maps signs to classes.
		Args:
			signs (np.array): 3 signs along axis 1

		Returns:
			python list of integers (classes)
		'''
        signs = list(map(tuple, signs))
        classes = operator.itemgetter(*signs)(self.classes_dict_inv)
        return classes

    def classes_to_signs(self, classes):
		'''
		Maps classes to signs.
		Args:
			classes (np.array): integer values for classes

		Returns:
			array of signs (np.array) with signs along axis 1
		'''
        signs = operator.itemgetter(*classes)(self.classes_dict)
        return signs


class SphericalManipulations():
	'''
	Various coordinate transformations.
	Angular Distance calculation.
	'''
	def __init__(self):
		pass

	def angles_to_cartesian_absvals_and_signs(self, angles):
		'''
		Transforms numpy array of azimuth and zenith
		to numpy arrays of signs like (-1, 1, -1) and
		L2 normalized absolute values like (1,0,0).

		Args:
			angles (np.ndarray): numpy array of (azi, zen) in rad

		Returns:
			Tuple of 2 np.arrays (np.ndarray, np.ndarray)
			array of signs. array of absolute values
		'''
		azi, zen = angles[:,0], angles[:,1]
		st = np.sin(zen)
		ct = np.cos(zen)
		sp = np.sin(azi)
		cp = np.cos(azi)
		# cartesian unit vec
		y1 = st * cp
		y2 = st * sp
		y3 = ct
		y_vec = np.column_stack([y1, y2, y3])
		# signs
		y_sign = np.sign(y_vec).astype('int')
		# positive spherical unit vector (abs vals)
		y_abs = y_vec / y_sign
		return y_sign, y_abs

	def cartesian_absvals_and_signs_to_angles(self, absvals, signs):
		'''
		Inverse of angles_to_cartesian_absvals_and_signs.
		'''
		dirvec = signs * absvals
		x,y,z = dirvec[:,0], dirvec[:,1], dirvec[:,2]
		azi = np.arctan2(y,x)
		idx = azi < 0
		azi[idx]+=2*np.pi

		arg = z / np.sqrt(x**2 + y**2 + z**2)
		arg = np.clip(arg, -0.999999, 0.999999)
		zen = np.arccos(arg)
		return np.column_stack([azi, zen])

	def angular_dist(self, y1, y2):
		'''
		calculate angular distance between y1 and y2
		y1 : vector of (az, zen)
		y2 : vector of (az, zen)
		'''
		sa1 = np.sin(y1[:, 0])
		ca1 = np.cos(y1[:, 0])
		sz1 = np.sin(y1[:, 1])
		cz1 = np.cos(y1[:, 1])

		sa2 = np.sin(y2[:, 0])
		ca2 = np.cos(y2[:, 0])
		sz2 = np.sin(y2[:, 1])
		cz2 = np.cos(y2[:, 1])

		scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
		return np.arccos(scalar_prod)


class SphericalExponentialActivationStable(Layer):
    '''
    Implements the exponential activation layer proposed Laio et al.
    in https://arxiv.org/pdf/1904.05404.pdf (2019), eq.9.
    '''
    def __init__(self, **kwargs):
        super(SphericalExponentialActivationStable, self).__init__(**kwargs)

    def call(self, x, axis=-1):
        '''
        Args:
            x (tf.tensor): inputs
            axis (int): axis to be normalized over
        Returns:
            res (tf.tensor): result of activations applied to inputs
        '''
        m = tf.math.reduce_max(x, axis, keepdims=True) # subtract maximum exponent
        z = tf.math.multiply(tf.exp(x-m), tf.exp(x-m)) # for numeric stability
        res = tf.exp(x-m) / tf.math.sqrt(tf.reduce_sum(z, axis, keepdims=True))
        return res


def get_model_predictions(model, x):
	'''
	Convert predictions from Spherical Regression model
	ala https://arxiv.org/pdf/1904.05404.pdf
	to zenith and azimuth angles. And keep track of
	max probability of the best octant classification.

	Args:
		model (tf.model): trained tensorflow model
		x (tf.tensor): model inputs

	Returns:
		directions and probabilities
		np.ndarray, float
	'''
	octc = OctantClasses()
	sphm = SphericalManipulations()

	absvals, logits = model.predict(x)
	probs = scipy.special.softmax(logits, axis=1)
	max_probs = np.max(probs, axis=1)
	classes = np.argmax(probs, axis=1)
	signs = np.array(octc.classes_to_signs(classes))

	angles = sphm.cartesian_absvals_and_signs_to_angles(absvals, signs)
	return angles, max_probs
