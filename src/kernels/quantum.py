from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit import Aer
from qiskit.quantum_info import Statevector
import qiskit as qiskit

import numpy as np
from concurrent import futures

NUM_REPS      = 1
PHASE_SCALING = 0.08
 
def calc_gram(X,Z):
	jobs_done = 0
	kern = np.zeros((X.shape[0], Z.shape[0]))
	ex = futures.ProcessPoolExecutor()
	jobs = []
	for i in range(0,X.shape[0]):
		for j in range(0,Z.shape[0]):
			x = X[i,:]
			z = Z[j,:]
			jobs.append(ex.submit(calc_kernel_dist,i,j,x,z))
	for f in futures.as_completed(jobs):
		r = f.result()
		kern[r[0],r[1]] = r[2]
		jobs_done = jobs_done+1
		print("============= JOBS_DONE: " + str(jobs_done) + "/" + str(len(jobs)) + " ================")
	
	return kern

def calc_gram_sym(X):
	jobs_done = 0
	kern = np.zeros((X.shape[0], X.shape[0]))
	ex = futures.ProcessPoolExecutor()
	jobs = []
	for i in range(0,X.shape[0]):
		for j in range(i,X.shape[0]):
			x1 = X[i,:]
			x2 = X[j,:]
			jobs.append(ex.submit(calc_kernel_dist,i,j,x1,x2))
	for f in futures.as_completed(jobs):
		r = f.result()
		kern[r[0],r[1]] = r[2]
		kern[r[1],r[0]] = r[2]
		jobs_done = jobs_done+1
		print("============= JOBS_DONE: " + str(jobs_done) + "/" + str(len(jobs)) + " ================")
	return kern

# Calculates the quantum kernel distance metric between two vectors.
# All elements in array should be in range [-pi, pi) as they will be encoded as phases in qubits.
def calc_kernel_dist(i, j, x_vec, z_vec):
	num_qubits = x_vec.shape[0]
	circ = QuantumCircuit(num_qubits)
	u_x  = ZZFeatureMap(num_qubits, reps=NUM_REPS)
	u_x  = u_x.bind_parameters(x_vec*PHASE_SCALING)
	circ = circ.compose(u_x)
	u_z  = ZZFeatureMap(num_qubits, reps=NUM_REPS).inverse()
	u_z  = u_z.bind_parameters(z_vec*PHASE_SCALING)
	circ = circ.compose(u_z)
	state = Statevector.from_int(0,2**num_qubits)
	state = state.evolve(circ)
	cnt = 0
	v = np.square(np.abs(state.data))
	val = v[0]
	return i, j, val
