import numpy as np
from scipy.spatial import ConvexHull
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from Vector_Comp import find_transf_matrix

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
with open(args.filename) as file:
	rDDF = pd.read_csv(file, skiprows=3)												  # Rows in CSV header
	print(rDDF.head(8))

# Start timer
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

df_log = pd.DataFrame(columns=['pts_1','pts_2','T_Matrix', 'nuScale_Orgin', 'nuScale','Error'])
for c in (hull_2.simplices):															  # Set of indicies of points forming second hull					  # For comparison (c) --> for simplex (s) --> for permutation (p)
	for s in (hull_1.simplices):														  # Set of indicies of points forming first hull
		for p in permutations(c,2):														  # Normal and flipped indicies for second hull points
			pts_2_t = pts_2 															  # reset our points to manipulate between checks
			#raw_input("Press Enter to continue...")
			pts_considered = np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]])
			print ('Pts_considered', pts_considered)
			transf_matrix = find_transf_matrix(np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]]))
			pts_2_t = transf_matrix.dot(np.c_[pts_2,np.zeros(np.size(pts_2,0)),np.ones(np.size(pts_2,0))].T) #https://mail.python.org/pipermail/python-list/2013-October/657294.html  --> apply the transformation matrix
			#pts_2_t_hull = transf_matrix.dot(np.c_[pts_2[hull_2.vertices],np.zeros(np.size(pts_2,0)),np.ones(np.size(pts_2[hull_2.vertices],0))].T)
			#print('hull_2_t', pts_2_t_hull.T[:,:2])
			side_consideration = np.array([hull_2.points[p[0]], hull_2.points[p[1]]])
			side_consideration_t = transf_matrix.dot(np.c_[side_consideration,np.zeros(np.size(side_consideration,0)),np.ones(np.size(side_consideration,0))].T)
			#print('side_consideration', side_consideration)
			#print('side_consideration_t', side_consideration_t.T[:,:2])
			print('pts_2_t', pts_2_t.T[:,:2])
			#print ('transf_matrix',transf_matrix)                    
			# X log permutation combination (indicies or otherwise)
			# calculate non-uniform scaling
			
				#angle of 2 relative to 1= atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
				#vectors ab1 & ac1 and ab2 & ac2
				#find angels theta1 and theta2
				#2dscale o = a1 & mag = (tan() - tan())

			nuScale = np.ones(3)														  # Scale values along X, Y, & Z axes
			# perform non uniform scaling (uScale)
			orgin = np.zeros(3)	         												  # Point serving as orgin for scale (point in face)
			#q = o * (1 - nuScale) + pts_2 * nuScale 									  # nuScaled set of points
			# calculate transformation
			# perform transformations
			# calculate error
			# log error for permutation
			df_log.loc[-1] = [np.array([pts_1[p[0]], pts_1[p[1]]]), np.array([pts_2[p[0]], pts_2[p[1]]]), transf_matrix, orgin, nuScale, 1]
			df_log.index = df_log.index + 1
			#print(p)

			RD1Fig = plt.figure()
			ax = RD1Fig.add_subplot(111)
			ax.plot(pts_1.T[0], pts_1.T[1], "ko")
			ax.plot(pts_2.T[0], pts_2.T[1], "bo")
			ax.plot(pts_2_t[0], pts_2_t[1], "gx") 										  # pts is already transposed due to the transformation matrix
			#ax.plot(q.T[0], q.T[1], "go")

			for s in hull_1.simplices:
				ax.plot(pts_1[s, 0], pts_1[s, 1], "r-")

			for s in hull_2.simplices:
				ax.plot(pts_2[s, 0], pts_2[s, 1], "r-")

			for pts_con in pts_considered:
				ax.plot(pts_con.T[0], pts_con.T[1], "y-")								  # is the program taking in one of the toupes as the x's and one as the y's?... yup needed to transpose it...

			ax.plot(side_consideration_t[:2,:][0], side_consideration_t[:2,:][1], "k-")

			ax.set_xlabel('X_1')
			ax.set_ylabel('Y_1')

			plt.show()

df_log = df_log.sort_index()  															  # sorting by index
print(df_log.head(12))
#End