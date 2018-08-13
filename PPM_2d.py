import numpy as np
from scipy.spatial import ConvexHull
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from Vector_Comp import find_transf_matrix
#from Transformation_Matrix import comp_matrix


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
			raw_input("Press Enter to continue...")  # input("Press Enter to continue...")
			pts_considered = np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]])
			print ('Pts_considered', pts_considered.shape)
			transf_matrix = find_transf_matrix(np.array([[hull_1.points[s[0]], hull_1.points[s[1]]], [hull_2.points[p[0]], hull_2.points[p[1]]]]))
			pts_2_t = transf_matrix.dot(np.c_[pts_2,np.zeros(np.size(pts_2,0)),np.ones(np.size(pts_2,0))].T) #https://mail.python.org/pipermail/python-list/2013-October/657294.html  --> apply the transformation matrix
			print('pts_2_t', pts_2_t.T[:,:2])
			#print ('transf_matrix',transf_matrix)                    
			# X log permutation combination (indicies or otherwise)
			# calculate non-uniform scaling
			
				#angle of 2 relative to 1= atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
				#vectors ab1 & ac1 and ab2 & ac2
				#find angels theta1 and theta2
				#2dscale o = a1 & mag = (tan() - tan())

			nuScale = np.array([1.5,.5,2])												  # Scale values along X, Y, & Z axes
			# perform non uniform scaling (uScale)
			o = pts_1[p[0]]	         													  # Point serving as orgin for scale (point in face)
			#q = o * (1 - nuScale) + pts_2 * nuScale 									  # nuScaled set of points
			# calculate transformation
			# perform transformations
			# calculate error
			# log error for permutation
			df_log.loc[-1] = [np.array([pts_1[p[0]], pts_1[p[1]]]), np.array([pts_2[p[0]], pts_2[p[1]]]), 0, o, nuScale, 1]
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

			ax.set_xlabel('X_1')
			ax.set_ylabel('Y_1')

			plt.show()

df_log = df_log.sort_index()  															  # sorting by index
print(df_log.head(12))



#for i, c in enumerate(hull_2.vertices):
#	if i == hull_2.vertices.shape[0]:
#		# Send current crd and next pt
#		h2_crd = np.matrix([hull_2.vertices[i],hull_2.vertices[-i]])
#	else:
#		# Send current crd and first pt
#		h2_crd = np.matrix([hull_2.vertices[i],hull_2.vertices[+1]])
#	for j, s in enumerate(hull_1.vertices):
#		if j == hull_1.vertices.shape[0]:
#			# Send current pt and next pt
#			h1_crd = np.matrix([hull_1.vertices[i],hull_1.vertices[-i]])
#		else:
#			# Send current pt and first pt
#			h1_crd = np.matrix([hull_1.vertices[i],hull_1.vertices[+1]])
#		print (h1_crd)
#		for h1_crd_p in permutations(h1_crd,2):
#			target_crd_set = np.vstack((h2_crd,h1_crd_p))
#			print ('target_crd_set', target_crd_set)