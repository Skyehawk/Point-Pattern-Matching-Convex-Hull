import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from Vector_Comp import find_transf_matrix

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
with open(args.filename) as file:
	rDDF = pd.read_csv(file, skiprows=3)												  # Rows in CSV header
	print(rDDF.head(8))

pts_1_df = rDDF[['X_1','Y_1']].as_matrix()
pts_1 = pts_1_df[np.logical_and(~np.isnan(pts_1_df)[:,0], ~np.isnan(pts_1_df)[:,1])]		  # Convert to np.matrix & filter for the scipy.convexhull
hull_1 = ConvexHull(pts_1)

pts_2_df = rDDF[['X_2','Y_2']].as_matrix()
pts_2 = pts_2_df[np.logical_and(~np.isnan(pts_2_df)[:,0], ~np.isnan(pts_2_df)[:,1])]
hull_2 = ConvexHull(pts_2)

pts_2_t = pts_2

print('pts_1', pts_1)
print('pts_2', pts_2)

print('Hull_1_pts',hull_1.points)
print('Hull_1_pts',hull_2.points)

df_log = pd.DataFrame(columns=['Matches', 'Total Error Squared','T_Matrix',])
start_time = timeit.default_timer()														  # start timer
for c in (hull_2.simplices):															  # Set of indicies of points forming second hull					  # For comparison (c) --> for simplex (s) --> for permutation (p)
	for s in (hull_1.simplices):														  # Set of indicies of points forming first hull
		for p in permutations(c,2):														  # Normal and flipped indicies for second hull points
			pts_2_t = pts_2 															  # reset our points to manipulate between checks
			#raw_input("Press Enter to continue...")
			pts_considered = np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]])
			#print ('Pts_considered', pts_considered)
			transf_matrix = find_transf_matrix(np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]]))
			pts_2_t = transf_matrix.dot(np.c_[pts_2,np.zeros(np.size(pts_2,0)),np.ones(np.size(pts_2,0))].T) #https://mail.python.org/pipermail/python-list/2013-October/657294.html  --> apply the transformation matrix
			side_consideration = np.array([hull_2.points[p[0]], hull_2.points[p[1]]])
			side_consideration_t = transf_matrix.dot(np.c_[side_consideration,np.zeros(np.size(side_consideration,0)),np.ones(np.size(side_consideration,0))].T)
			#print('pts_2_t', pts_2_t.T[:,:2])

			aligned_pts = np.vstack((pts_1, pts_2_t.T[:,:2]))
			#print('aligned_pts', aligned_pts)
			pts_tree = cKDTree(aligned_pts)
			rows_to_fuse = pts_tree.query_pairs(r=0.1)									  # TODO: We need to create a dynamic threshold
			#print(repr (list(rows_to_fuse)))
			#print(repr(aligned_pts[list(rows_to_fuse)]))

			err_sq = 0.																	  # err^2 accumulator
			for idx_touple in rows_to_fuse:
				if idx_touple[1] + 1 > np.size(pts_1,axis=0) and idx_touple[0] + 1 <= np.size(pts_1,axis=0):	  # TODO: check to make sure pts are not both in pts2_t as well
					err_sq += (aligned_pts[idx_touple[0]][0]-aligned_pts[idx_touple[1]][0])**2 + (aligned_pts[idx_touple[0]][1]-aligned_pts[idx_touple[1]][1])**2
			
			#print ('transf_matrix',transf_matrix)                    
			# X log permutation combination (indicies or otherwise)
			# calculate non-uniform scaling

			nuScale = np.ones(3)														  # Scale values along X, Y, & Z axes
			# perform non uniform scaling (uScale)
			#q = o * (1 - nuScale) + pts_2 * nuScale 									  # nuScaled set of points
			# calculate transformation
			# perform transformations
			# calculate error
			# log error for permutation
			df_log.loc[-1] = [len(rows_to_fuse), err_sq, transf_matrix]
			df_log.index = df_log.index + 1
			#print(p)

			#RD1Fig = plt.figure()
			#ax = RD1Fig.add_subplot(111)
			#ax.plot(pts_1.T[0], pts_1.T[1], "ko")
			#ax.plot(pts_2.T[0], pts_2.T[1], "bo")
			#ax.plot(pts_2_t[0], pts_2_t[1], "gx") 										  # pts is already transposed due to the transformation matrix
			##ax.plot(q.T[0], q.T[1], "go")

			#for s in hull_1.simplices:
			#	ax.plot(pts_1[s, 0], pts_1[s, 1], "r-")

			#for s in hull_2.simplices:
			#	ax.plot(pts_2[s, 0], pts_2[s, 1], "r-")

			#for pts_con in pts_considered:
			#	ax.plot(pts_con.T[0], pts_con.T[1], "y-")

			#ax.plot(side_consideration_t[:2,:][0], side_consideration_t[:2,:][1], "k-")

			#ax.set_xlabel('X_1')
			#ax.set_ylabel('Y_1')

			#plt.show()
elapsed = timeit.default_timer() - start_time
print('------------------------------------')
print('Execution Time: ', elapsed, 'sec.')
df_log = df_log.sort_values(['Matches', 'Total Error Squared'], ascending=[False, True])  # sorting by nnumber of matches then by err^2
#df_log = df_log.sort_index()  															  # sorting by index
print(df_log.head(12))
#End