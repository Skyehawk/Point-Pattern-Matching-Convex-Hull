import numpy as np
from Transformation_Matrix import comp_matrix

# find what it takes to transform v1 to v0, return composed translation matrix

translation_offset = np.empty((2), dtype=float)											  # Needs to be upped to 3 for 3d

def find_scale (pts):																	  # TODO: check for accuracy, close, but off in testing
	global translation_offset
	dist_0 = 0
	dist_1 = 0
	if (pts.shape == (2,2,3)):
		dist_0 = np.sqrt((pts[0,1,0]-pts[0,0,0])**2 + (pts[0,0,1]-pts[0,1,1])**2 + (pts[0,1,2]-pts[0,0,2])**2)	# Length of first simplex (3d)
		dist_1 = np.sqrt((pts[1,1,0]-pts[1,0,0])**2 + (pts[1,1,1]-pts[1,0,1])**2 + (pts[1,1,2]-pts[1,0,2])**2)	# Length of second simplex (3d)
		scale_factor = (dist_1/dist_0)
		translation_offset += np.append((1/scale_factor)*pts[0,0],0)
		return (dist_1/dist_0)*np.ones(3)
	else: 
		dist_0 = np.sqrt((pts[0,1,0]-pts[0,0,0])**2 + (pts[0,0,1]-pts[0,1,1])**2)		  # Length of first simplex (2d)
		dist_1 = np.sqrt((pts[1,1,0]-pts[1,0,0])**2 + (pts[1,1,1]-pts[1,0,1])**2)		  # Length of second simplex (2d)
		scale_factor = (dist_1/dist_0)
		translation_offset += ((1/scale_factor)*pts[0,0]) - pts[0,0]
		#print('target_pt', pts[0,0])
		#print('scale_factor', dist_1/dist_0)
		#print('translation_offset',translation_offset)
		return (dist_1/dist_0)*np.ones(2)
	
def find_rot (pts):
	global translation_offset															  # access to the translation accumulator (still need to implement calc of offset)
	v0 = pts[0,1] -	pts[0,0]															  # v0 = [x,y,z] = [i<hat>, j<hat>, k<hat>]
	v1 = pts[1,1] - pts[1,0]															  # v1 = [x,y,z] = [i<hat>, j<hat>, k<hat>]
	#print('rot_v', v0, v1)																
	alpha = np.arctan2(v1[1],v1[0]) - np.arctan2(v0[1],v0[0])							  # alpha angle about x-axis of standard basis (rad)
	r = np.sqrt(pts[0,0,0]**2 + pts[0,0,1]**2)
	rot_off = np.array([np.cos(alpha) * r , np.sin(alpha) * r])
	#print ('rot_off', rot_off)
	#print ('rot_pt_off', pts[0,0])
	#print ('alpha r', alpha, r)
	temp_offset = (pts[0,0] - rot_off)													  # failing due to rot_off sometimes being larger than the value of the pt it is being subtracted from

	#find quadrent & apply based on that for offset
	if rot_off[0]>0 and rot_off[1]>0:													  # q1
		temp_offset *= np.array([1,-1])
	elif rot_off[0]<0 and rot_off[1]>0:													  # q2
		temp_offset *= np.array([-1,-1])
	elif rot_off[0]<0 and rot_off[1]<0:													  # q3
		temp_offset *= np.array([-1,1])
	elif rot_off[0]>0 and rot_off[1]<0:													  # q4
		temp_offset *= np.array([1,1])													  # no change
	elif rot_off[0]>0 and rot_off[1]==0:													  # x+
		temp_offset *= np.array([1,1])
	elif rot_off[0]>0 and rot_off[1]==0:													  # x-
		temp_offset *= np.array([1,1])
	elif rot_off[0]==0 and rot_off[1]>0:													  # y+
		temp_offset *= np.array([-1,-1])
	elif rot_off[0]==0 and rot_off[1]<0:													  # y-
		temp_offset *= np.array([-1,-1])
	elif rot_off[0]==0 and rot_off[1]==0:												  # orgin
		print ('WARNING: point at orgin in rotation offset, should only occur if point was at orgin initially (0,0,0) offset required')
	translation_offset += temp_offset

	if (pts.shape == (2,2,3)):
		beta = np.arctan2(v1[2], np.sqrt(np.pow(v1[0],2) + np.pow(v1[1],2))) 			  # beta angle about y-axis of standard basis (rad)
		- np.arctan2(v0[2], np.sqrt(np.pow(v0[0],2) + np.pow(v0[1],2)))
		return  np.array([0,-beta,-alpha])												  # todo: fix the locations of these, rotation about the: {x-axis, y-axis, z-axis}
		#return np.array([0,0,0])
	return np.array([0,0,-alpha])
	#return np.array([0,0,0])															  # Testing of proper roation applicatin by the TM algorithm (rad)

def find_shear (pts):																	  # to do: build in shear support (warp support) for 3d cases
	return np.ones(3)

def find_trans (pts):																	  # todo: add in the translation_offset for rotation & scale
	global translation_offset
	if (pts.shape == (2,2,3)):
		return -(np.append(pts[1,0] - pts[0,0],0)) + translation_offset 
	return -(pts[1,0] - pts[0,0]) + translation_offset 
	
def find_transf_matrix (pts):															  # Do a check that we are getting 2 or 3 dimentional crds
	print('pts', pts)
	global translation_offset															  # Create a touple with correct dimentions bsed on input pts
	if (pts.shape == (2,2,3)):
		translation_offset = np.append(np.zeros((2)),0)	
	elif (pts.shape == (2,2,2)):
		translation_offset = np.zeros((2))
	else : 
		print ('Dimentions of input crds not in 2d or 3d')								  # todo: throw exception	
	return (comp_matrix( find_scale(pts), find_rot(pts), find_shear(pts), find_trans(pts)))
	