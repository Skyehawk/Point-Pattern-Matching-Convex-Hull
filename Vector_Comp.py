import numpy as np 																		  # ver. 1.15.0
import scipy 																			  # ver. 1.0.0
from Transformation_Matrix import comp_matrix 											  # ver. 0.1.0

#** 
# @author      Skye Leake <skleake96@gmail.com>
# @version     0.1.0
# @since       0.0.0
#/
translation_offset = np.empty((2), dtype=float)											  # Accumulator for tranlation offset from other transformations; Needs to be upped to 3 for 3d, we can handle this in the find transf_matrix() method
uniform_scale_factor = 1																  # Uniform scaling factor

#**
# Find scaling factor between two edges, second edge is matched to first
# @param  pts 			 - [[float,float,(float)],[float,float,(float)]] (-inf,inf) Input points defining edge1 and edge2
# @return scale_array	 - [float,float,(float)]
#/
def find_scale (pts):
	global translation_offset
	global uniform_scale_factor
	dist_0 = 0
	dist_1 = 0
	if (pts.shape == (2,2,3)):
		dist_0 = np.sqrt((pts[0,1,0]-pts[0,0,0])**2 + (pts[0,1,1]-pts[0,0,1])**2 + (pts[0,1,2]-pts[0,0,2])**2)	# Length of first simplex (3d)	for first vector: sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
		dist_1 = np.sqrt((pts[1,1,0]-pts[1,0,0])**2 + (pts[1,1,1]-pts[1,0,1])**2 + (pts[1,1,2]-pts[1,0,2])**2)	# Length of second simplex (3d)
		uniform_scale_factor = (dist_0/dist_1)
		scale_factor = np.array([pts[0,0,0]/pts[1,0,0], pts[0,0,1]/pts[1,0,1], pts[0,0,2]/pts[1,0,2]])
		return uniform_scale_factor*np.ones(3)
	else: 
		dist_0 = np.sqrt((pts[0,1,0]-pts[0,0,0])**2 + (pts[0,1,1]-pts[0,0,1])**2)		  # Length of first simplex (2d)	for first vector: sqrt((x2-x1)**2+(y2-y1)**2)
		dist_1 = np.sqrt((pts[1,1,0]-pts[1,0,0])**2 + (pts[1,1,1]-pts[1,0,1])**2)		  # Length of second simplex (2d)
		uniform_scale_factor = (dist_0/dist_1)
		return uniform_scale_factor*np.ones(2)
	
#**
# Find alpha and beta angles between two edges, second edge is matched to first, from x+ (right handed)
# @param  pts 				- [[float,float,(float)],[float,float,(float)]] (-inf,inf) Input points defining edge1 and edge2
# @return rotation_array 	- [float,float,(float)] (radians)
#/	
def find_rot (pts):
	global translation_offset															  # Access to the translation accumulator
	global uniform_scale_factor
	v0 = pts[0,0] -	pts[0,1]															  # TODO: phase out v0 & v1 if posssible		# v0 = [x,y,z] = [i<hat>, j<hat>, k<hat>]
	v1 = pts[1,0] - pts[1,1]															  												# v1 = [x,y,z] = [i<hat>, j<hat>, k<hat>]

	alpha = np.arctan2(v1[1],v1[0]) - np.arctan2(v0[1],v0[0])							  # Alpha angle about x-axis of standard basis (rad)
	r = np.sqrt(pts[1,0,0]**2 + pts[1,0,1]**2)											  

	qx = np.cos(-alpha) * pts[1,0,0] - np.sin(-alpha) * pts[1,0,1]
	qy = np.sin(-alpha) * pts[1,0,0] + np.cos(-alpha) * pts[1,0,1]
	s2_align_pt = np.array([qx,qy])														  # Calculate position of second alignment point after rotation of -alpha about (0,0,0)
	translation_offset += (- (s2_align_pt * uniform_scale_factor) + pts[0,0])			  # Find differance between found alignment point and alignment point on first edge

	if (pts.shape == (2,2,3)):
		beta = np.arctan2(v1[2], np.sqrt(np.pow(v1[0],2) + np.pow(v1[1],2))) 			  # Beta angle about y-axis of standard basis (rad)
		- np.arctan2(v0[2], np.sqrt(np.pow(v0[0],2) + np.pow(v0[1],2)))
		return  np.array([0,-beta,-alpha])												  # TODO: fix the locations of these, rotation about the: {x-axis, y-axis, z-axis}
	return np.array([0,0,-alpha])	

#**
# Find shearing factor between two edges, second edge is matched to first
# @param  pts 			- [[float,float,(float)],[float,float,(float)]] (-inf,inf) Input points defining edge1 and edge2
# @return shear_array	- [float,float,(float)]	(Not implemented)
#/
def find_shear (pts):																	  # TODO: build in shear support (warp support) for 3d cases
	return np.ones(3)

#**
# Find (apply) translation factor between two edges, second edge is matched to first, first point in edge is used as alignment pt
# @param  pts 				- [[float,float,(float)],[float,float,(float)]] (-inf,inf) Input points defining edge1 and edge2
# @return translation_array - [float,float,(float)] (-inf,inf)	translation along x,y,z axes
#/
def find_trans (pts):																	  # TODO: add in the translation_offset for rotation & scale
	global translation_offset
	if (pts.shape == (2,2,3)): 
		return translation_offset 														  # TODO: test in 3D case
	return translation_offset

#**
# Find transforation matrix to bring two vectors (edges) into alignment, second vector is matched to first
# @param  pts 			- [[float,float,(float)],[float,float,(float)]] (-inf,inf) Input points defining edge1 and edge2
# @return scaling_array - [floats] 4x4 transformation matrix
#/
def find_transf_matrix (pts):															  # Do a check that we are getting 2 or 3 dimentional crds
	global translation_offset															  # Create a touple with correct dimentions bsed on input pts
	if (pts.shape == (2,2,3)):
		translation_offset = np.append(np.zeros((2)),0)	
	elif (pts.shape == (2,2,2)):
		translation_offset = np.zeros((2))
	else : 
		print ('Dimentions of input crds not in 2d or 3d')								  # todo: throw exception 
	return (comp_matrix(find_scale(pts), find_rot(pts), find_shear(pts), find_trans(pts)))
	