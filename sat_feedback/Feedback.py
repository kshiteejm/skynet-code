from Input import *
from z3 import *
from Policies import *
import random


class Feedback(object):
	def __init__(self):
		input = Input()
		num_policies = len(input.getPols())
		# create z3 object with all arguments
		self.z3Solver = Solver()
		self.z3Solver.set(unsat_core=True)
		#self.z3Solver.set(':smt.phase_selection', 5)
		#self.z3Solver.set(':smt.random_seed', random.randint(1, 1000))
		self.edges = dict()
		self.reverseEdges = dict()
		# create adjacency matrix
		self.createEdgeSet(input.getTopo())

		# create sat variables for policies
		self.initializeSATVariables(num_policies)

		# Constraint Stores
		self.backwardReachPropagationConstraints = dict()

		# BFS Global Variable.
		self.bfsLists = dict()
		for i in range(1, len(self.edges) + 1) :
			self.bfsLists[i] = []

		# create policy constraints
		curpol = 0
		for pol in input.getPols():
			if pol[0] == Policies.reachability:
				self.addReachabilityConstraints(curpol, pol[1], pol[2])
			elif pol[0] == Policies.waypoint:
				self.addReachabilityConstraints(curpol, pol[1], pol[2], pol[3])
			elif pol[0] == Policies.isolation:
				self.addIsolationConstraints(pol[1], pol[2])
			curpol = curpol + 1

		print self.z3Solver

		self.z3Solver = Optimize()
		modelsat = self.z3Solver.check()

		#self.z3solveTime += time.time() - solvetime
		if modelsat == z3.sat :
			fwdmodel = self.z3Solver.model()
			print fwdmodel


	# add reachability and waypoint constraints	
	def addReachabilityConstraints(self, pol, srcSw, dstSw, W=None, pathlen=0) :
		#reachtime = time.time()
		if pathlen == 0 :
			# Default argument. Set to max.
			pathlen = len(self.edges)
		
		# Add Reachability in atmost pathlen steps constraint. 
		#reachAssert = self.Reach(dstSw,pc,pathlen) == True

		# src is reachable
		#self.reachvars[srcSw][pol][0] = True
		# Destination is reachable in <= plen steps
		reachAssertions = []
		for plen in range(1,pathlen+1) :
			reachAssertions.append(self.reachvars[dstSw][pol][plen])

		reachAssert = Or(*reachAssertions)

		#addtime = time.time() # Profiling z3 add.
		self.z3Solver.add(reachAssert)
		#self.z3addTime += time.time() - addtime

		# At Destination, forwarding has to stop here. So, F(d,neighbour(d),pc,1) == False 
		# When we perform the translation to rules, we can forward it to host accordingly.
		neighbours = self.edges[dstSw]
		
		destAssert = True
		for n in neighbours :
			destAssert = And(destAssert, self.fwdvars[dstSw][n][pol] == False)

		#addtime = time.time() # Profiling z3 add.
		self.z3Solver.add(destAssert)
		#self.z3addTime += time.time() - addtime

		#'''
		# If waypoints, add ordered waypoint constraints.
		if W <> None and len(W) > 0 : 
			totalwaypointCount = 0
			currwaypointCount = 0
			#for wayptSet in W :
			#	totalwaypointCount += len(wayptSet)
			totalwaypointCount = len(W)

			prevWayptSet = None
			for wayptSet in W : 
				# ordered Waypoints.
				# Add the Waypoint Constraints. 
				#currwaypointCount += len(wayptSet)
				#for w in wayptSet :
				w = wayptSet			
				reachAssertions = []
				#for plen in range(1 + currwaypointCount - len(wayptSet), pathlen - (totalwaypointCount - currwaypointCount)) :
				for plen in range(1, pathlen + 1):
					reachAssertions.append(self.reachvars[w][pol][plen])

					if prevWayptSet <> None : 
						#for w2 in prevWayptSet :
						w2 = prevWayptSet 
						orderAssertions = []
						for plen2 in range(1, plen): 
							orderAssertions.append(self.reachvars[w2][pol][plen2])
						orderAssert = Implies(self.reachvars[w][pol][plen], Or(*orderAssertions))
						self.z3Solver.add(orderAssert)
				
				reachAssert = Or(*reachAssertions)

				#self.z3numberofadds += 1
				#addtime = time.time() # Profiling z3 add.
				self.z3Solver.add(reachAssert)
				#self.z3addTime += time.time() - addtime
				prevWayptSet = wayptSet

		#st = time.time()
		#'''
		# This prunes the Reach relation to only valid 
		# states by constructing a tree from the source switch
		# For example, if a switch sw is a distance = 3 from src, then 
		# Reach(sw,pc,[0,1,2]) is trivially False.
		self.addTopologyTreeConstraints(srcSw, pol)

		# Add Path Constraints for this flow to find the forwarding model for this flow.
		self.addPathConstraints(srcSw,pol)		


	# add isolation constraints	
	def addIsolationConstraints(self, pc1, pc2):
		for sw in self.edges:
			for n in self.edges[sw]:
				isolateAssert = Not( And (self.fwdvars[sw][n][pc1], self.fwdvars[sw][n][pc2]))
				#self.z3numberofadds += 1
				#addtime = time.time() # Profiling z3 add.
				self.z3Solver.add(isolateAssert)	

	# add constraints for path
	def addPathConstraints(self, src, pc) :
		swCount = len(self.edges)
		maxPathLen = swCount

		#print self.edges

		neighbours = self.edges[src]
		
		srcAssertions = []
		for n in neighbours : 
			srcAssertions.append(And(self.fwdvars[src][n][pc], self.reachvars[n][pc][1]))

		#addtime = time.time() # Profiling z3 add.
		self.z3Solver.add(Or(*srcAssertions))
		#self.z3addTime += time.time() - addtime

		#st = time.time()
		#constime = 0
		#addtime = 0
		for i in self.edges :
			if i == src : 
				continue

			for pathlen in range(1,maxPathLen+1) :
				if i not in self.bfsLists[pathlen] : 
					# Not at distance i in the topology tree, dont add constraints.
					continue 

				ineighbours = self.reverseEdges[i]
				
				# Backward reachability proogation constraints.  
				# If a node $n_1$ is reachable in $k$ steps, there must be a node $n_2$ reachable in  $k-1$ steps and 
				# a forwarding rule $n_2 \rightarrow n_1$.

				constraintKey = str(i) + ":" + str(pc) + "*" + str(pathlen)
				if constraintKey in self.backwardReachPropagationConstraints : 
					# Reuse constraint object if already created.
					backwardReachConstraint = self.backwardReachPropagationConstraints[constraintKey]
				else : 
					# Create constraint.
					beforeHopAssertions = []

					#beforeHopAssertionsStr = ""
					#ct = time.time()
					for isw in ineighbours : 
						#print isw, pc, pathlen - 1
						#print self.reachvars[isw][pc][pathlen - 1]
						if isw == src:
							beforeHopAssertions.append(And(self.fwdvars[isw][i][pc], True))	
						else:
							if pathlen == 1:
								beforeHopAssertions.append(And(self.fwdvars[isw][i][pc], False))	
							else:
								beforeHopAssertions.append(And(self.fwdvars[isw][i][pc], self.reachvars[isw][pc][pathlen - 1]))
						
					backwardReachConstraint = Implies(self.reachvars[i][pc][pathlen], Or(*beforeHopAssertions))
					#constime += time.time() - ct

				#at = time.time()	
				self.z3Solver.add(backwardReachConstraint)
				#addtime += time.time() - at

				# Store constraint for reuse. 
				constraintKey = str(i) + ":" + str(pc) + "*" + str(pathlen)
				if constraintKey not in self.backwardReachPropagationConstraints :
					self.backwardReachPropagationConstraints[constraintKey] = backwardReachConstraint


	# Construct a topology tree for each packet class from src to
	# prune the Reach relation for unreachable switches""" 
	def addTopologyTreeConstraints(self, srcSw, pc) : 

		swCount = len(self.edges)
		maxPathLen = swCount

		swList = [srcSw]
		for k in range(1, maxPathLen + 1) :
			newSwList = []
			for sw in swList :
				neighbours = self.edges[sw]
				for n in neighbours :
					if n not in newSwList : 
						newSwList.append(n)

			self.bfsLists[k] = newSwList
			# Set switches not in newSwList to false at Reach(k)
			for sw in self.edges :
				if sw not in newSwList  :
					self.z3Solver.add(Not(self.reachvars[sw][pc][k]))

			swList = newSwList
	

	# find out all the valid edges from the topology matrix
	def createEdgeSet(self, topo):
		for i in range(len(topo)):
			self.edges[i] = list()
			self.reverseEdges[i] = list()

		for i in range(len(topo)):			
			for j in range(len(topo[i])):
				if topo[i][j] == 1:
					self.edges[i].append(j)
					self.reverseEdges[j].append(i)
		#return validEdge

	#def getNumNodes(validEdge):


	def initializeSATVariables(self, num_policies):
		swCount = len(self.edges)
		#pcRange = self.pdb.getPacketClassRange()
		#maxPathLen = self.topology.getMaxPathLength()

		maxPathLen = swCount

		self.fwdvars = dict()
		self.reachvars = dict()

		for sw1 in self.edges:
			self.fwdvars[sw1] = dict()
			for sw2 in self.edges[sw1]:
				self.fwdvars[sw1][sw2] = dict()
				for pc in range(0, num_policies) :
					self.fwdvars[sw1][sw2][pc] = Bool("s"+str(sw1)+"-"+"s"+str(sw2)+":"+str(pc))
					#self.z3Solver.add(self.fwdvars[sw1][sw2][pc])

		for sw in self.edges:
			self.reachvars[sw] = dict()
			for pc in range(0, num_policies) :
				self.reachvars[sw][pc] = dict()
				for plen in range(1,maxPathLen +1) :
					self.reachvars[sw][pc][plen] = Bool("s"+str(sw)+":pol"+str(pc)+":"+str(plen))
					#self.z3Solver.add(self.reachvars[sw][pc][plen])
  
if __name__== "__main__":
  feedback = Feedback()
