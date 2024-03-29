import argparse
import timeit
import numpy as np 																		  # ver. 1.15.0
from scipy.spatial import ConvexHull, cKDTree											  # ver. 1.0.0
import pandas as pd 																	  # ver. 0.21.1
import matplotlib.pyplot as plt 														  # ver. 2.2.2
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import multiprocessing as mp
import tqdm

from Vector_Comp import find_transf_matrix												  # ver. 0.1.0 
from Transformation_Matrix import comp_matrix, decomp_matrix							  # ver. 0.1.0 only used in this file for the generation of test data
from sklearn.datasets import make_blobs

parser = argparse.ArgumentParser()														  # Take a path to the .cv file containing the points as input
#parser.add_argument("-i", "--input", help=" path to the input file (GRIB)")
parser.add_argument("-f", "--filename", help=" path to the input file (.csv)")
args = vars(parser.parse_args())

# --- Define utility functions ---
def solve_match_instance(d, index, hull_1, hull_2, pts_1, pts_2, c, s, p):
	pts_2_t = pts_2 															  # reset our points to manipulate between checks
	#raw_input("Press Enter to continue...")
	pts_considered = np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]])
	transf_matrix = find_transf_matrix(np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]]))
	pts_2_t = transf_matrix.dot(np.c_[pts_2[:,:2],np.zeros(np.size(pts_2,0)),np.ones(np.size(pts_2,0))].T) #https://mail.python.org/pipermail/python-list/2013-October/657294.html  --> apply the transformation matrix
	side_consideration = np.array([hull_2.points[p[0]], hull_2.points[p[1]]])
	side_consideration_t = transf_matrix.dot(np.c_[side_consideration,np.zeros(np.size(side_consideration,0)),np.ones(np.size(side_consideration,0))].T)
	#print('pts_2_t', pts_2_t.T[:,:2])

	aligned_pts = np.vstack((pts_1, pts_2_t.T[:,:2]))
	pts_tree = cKDTree(aligned_pts)												  # create a KD tree to quickly find pairs f points falling w/in a threshold distance
	rows_to_fuse = pts_tree.query_pairs(r=0.000001)								  # TODO: We need to create a dynamic threshold

	err_sq = 0.																	  # err^2 accumulator
	for idx_touple in rows_to_fuse:
		if idx_touple[1] + 1 > np.size(pts_1,axis=0) and idx_touple[0] + 1 <= np.size(pts_1,axis=0):	  # TODO: check to make sure pts are not both in pts2_t as well
			err_sq += (aligned_pts[idx_touple[0]][0]-aligned_pts[idx_touple[1]][0])**2 
			+ (aligned_pts[idx_touple[0]][1]-aligned_pts[idx_touple[1]][1])**2

	d[index] = [len(rows_to_fuse), err_sq, transf_matrix]					  # log our parameters


def main():
	manager = mp.Manager()
	results = manager.dict()
	pool = mp.Pool(12)

	jobs = []

	pts_1 = np.zeros(np.shape([1,1]))
	pts_2 = np.zeros(np.shape([1,1]))

	if args["filename"]:
		print('Reading Data...')
		with open(str(args["filename"])) as file:
			rDDF = pd.read_csv(file, skiprows=3)												  # Rows in .csv header
			print(rDDF.head(8))


		pts_1_df = rDDF[['X_1','Y_1']].to_numpy()
		pts_1 = pts_1_df[np.logical_and(~np.isnan(pts_1_df)[:,0], ~np.isnan(pts_1_df)[:,1])]	  # Convert to np.matrix & filter to remove nulls
		hull_1 = ConvexHull(pts_1)

		pts_2_df = rDDF[['X_2','Y_2']].to_numpy()
		pts_2 = pts_2_df[np.logical_and(~np.isnan(pts_2_df)[:,0], ~np.isnan(pts_2_df)[:,1])]	  # Convert to np.matrix & filter to remove nulls
		hull_2 = ConvexHull(pts_2)

		pts_2_t = pts_2

		#print('pts_1', pts_1)
		#print('pts_2', pts_2)

	else:
		print('Generating Test Data...')
		pts_1, y = make_blobs(n_samples=1000, centers=1, n_features=2, cluster_std=1.0, random_state=None)
		#print(pts_1)																		  # Generate some test data if input file is not supplied
		rndTransParams = np.random.rand(3,3)												  # Create a random transformation
		print("Random trans Params: \n" + 
			"\n  Scale: " + str(rndTransParams[0,0]*2) + "\n  Rotation: " + str(np.array([0,0,rndTransParams[1,0]*2*np.pi])) +
			"\n  Shear: " + str(np.ones(3)) + "\n  Translation: " + str(rndTransParams[2,:2]))
		
		tdata_transf_matrix = comp_matrix(np.array([rndTransParams[0,0]*2, rndTransParams[0,0]*2]), np.array([0,0,rndTransParams[1,0]*2*np.pi]), np.ones(3), rndTransParams[2,:2])
		pts_2 = tdata_transf_matrix.dot(np.c_[pts_1,np.zeros(np.size(pts_1,0)),np.ones(np.size(pts_1,0))].T)
		pts_2 = pts_2.T[:250,:2]															  # hold part of the dataset back to simulate differance

	hull_1 = ConvexHull(pts_1)
	hull_2 = ConvexHull(pts_2)
	
	print('Starting Solve...')
	start_time = timeit.default_timer()														  # Start timer
	i = 0																					  # count for idx of results
	for c in hull_2.simplices:												 				  # Set of indicies of points forming second hull
		for s in hull_1.simplices:														  	  # Set of indicies of points forming first hull
			for p in permutations(c,2):														  # Normal and flipped indicies for second hull points
				job = pool.apply_async(solve_match_instance, (results,i,hull_1,hull_2,pts_1,pts_2,c,s,p))
				jobs.append(job)
				i +=1

	for job in jobs:#tqdm(jobs,desc="Solving Data Alignment"):
		job.get()

	elapsed = timeit.default_timer() - start_time												   # Stop timer & get duration
	
	pool.close()
	pool.join()

	columns=['Matches', 'Total Error Squared','T_Matrix']
	resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)
	resultsDF.sort_values(['Matches', 'Total Error Squared'], ascending=[False, True], inplace=True)  # sorting by number of matches then by err^2
	resultsDF.reset_index(inplace=True, drop=True)

	print('------------------------------------')
	print('Execution Time: ', elapsed, 'sec.')
	print ("Iter: ", i)
	print("\nBest Match Transformation Matrix To Convert Set \"B\" To Set \"A\":\n" + str(resultsDF.at[0,'T_Matrix']))

	print("\n", resultsDF[['Matches','Total Error Squared']].head(12))

	pts_2_t = resultsDF.at[0,'T_Matrix'].dot(np.c_[pts_2,np.zeros(np.size(pts_2,0)),np.ones(np.size(pts_2,0))].T) 	  # apply closest match's transformation parameters

	RD1Fig = plt.figure()
	ax = RD1Fig.add_subplot(111)

	for s in hull_1.simplices:
		ax.plot(pts_1[s, 0], pts_1[s, 1], "r-")
	#for s in hull_2.simplices:
	#	ax.plot(pts_2[s, 0], pts_2[s, 1], "r-")
	#for pts_con in pts_considered:
	#	ax.plot(pts_con.T[0], pts_con.T[1], "y-")
	#ax.plot(side_consideration_t[:2,:][0], side_consideration_t[:2,:][1], "k-")

	ax.plot(pts_1.T[0], pts_1.T[1], "ko")
	#ax.plot(pts_2.T[0], pts_2.T[1], "gx")
	ax.plot(pts_2_t[0], pts_2_t[1], "cP") 													  # pts is already transposed due to the transformation matrix
	#ax.plot(q.T[0], q.T[1], "go")

	ax.set_xlabel('X_1')
	ax.set_ylabel('Y_1')

	plt.show()

if __name__ == '__main__':
	main()
#End