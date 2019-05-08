#import numpy as np

#A = np.array([[1, 2, 3], [3, 4, 5]])

from Policies import *

class Input(object):
	def __init__(self):
		# 5 switches
		# represents adjacency matrix of topology as list of lists
		self.topo = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]

		# 3 flows
		self.partial_routing = [[0, 1, 2], [0]]

		'''
		need info of exact path
		'''

		#3 policies
		# format of each policy <policyname, [arguments]>
		self.pols = [[Policies.reachability, 0, 3], [Policies.waypoint, 0, 4, [1]], [Policies.isolation, 0, 1]]

	# return adjacency matrix of topology
	def getTopo(self):
		return self.topo

	# return policies
	def getPols(self):
		return self.pols

	# return partial routes of all policies
	def getPartial(self):
		return self.partial_routing

