import numpy as np
from scipy.spatial import ConvexHull
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from Transformation_Matrix import comp_matrix


parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
with open(args.filename) as file:
	rDDF = pd.read_csv(file, skiprows=3)
	print(rDDF.head())

# Start timer

pts_1 = rDDF[['X_1','Y_1','Z_1']].as_matrix()											  # Convert to np.matrix for the scipy.convexhull
hull_1 = ConvexHull(pts_1)
pts_2 = rDDF[['X_2','Y_2','Z_2']].as_matrix()
hull_2 = ConvexHull(pts_2)

#print(hull_1.simplices)
#print(hull_2.simplices)

df_log = pd.DataFrame(columns=['Pts_1','Pts_2','T_Matrix', 'nuScale_Orgin', 'nuScale','Error'])
for c in (hull_2.simplices):															  # for comparison (c) --> for simplex (s) --> for permutation (p)
	for s in (hull_1.simplices):														  # All possible pairs of indecies of each face in cvx hull
		for p in permutations(s,3):                             
			# X log permutation combination (indicies or otherwise)
			# calculate non-uniform scaling
			
				#angle of 2 relative to 1= atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
				#vectors ab1 & ac1 and ab2 & ac2
				#find angels theta1 and theta2
				#2dscale o = a1 & mag = (tan() - tan())

			nuScale = np.array([1.5,.5,2])												  # Scale values along X, Y, & Z axes
			# perform non uniform scaling (uScale)
			o = pts_1[p[0]]	         													  # Point serving as orgin for scale (point in face)
			q = o * (1 - nuScale) + pts_2 * nuScale 									  # nuScaled set of points
			# calculate transformation
			# perform transformations
			# calculate error
			# log error for permutation
			df_log.loc[-1] = [np.array([pts_1[p[0]], pts_1[p[1]], pts_1[p[2]]]), np.array([pts_2[c[0]], pts_2[c[1]], pts_2[c[0]]]), 0, o, uScale, 1]
			df_log.index = df_log.index + 1
			#print(p)
df_log = df_log.sort_index()  															  # sorting by index
print(df_log.head(12))

RD1Fig = plt.figure()
ax = RD1Fig.add_subplot(111, projection="3d")
ax.plot(pts_1.T[0], pts_1.T[1], pts_1.T[2], "ko")
ax.plot(q.T[0], q.T[1], q.T[2], "go")

for s in hull_1.simplices:
    s = np.append(s, s[0])  															  # Here we cycle back to the first coordinate
    ax.plot(pts_1[s, 0], pts_1[s, 1], pts_1[s, 2], "r-")

ax.set_xlabel('X_1')
ax.set_ylabel('Y_1')
ax.set_zlabel('Z_1')

plt.show()
