import numpy as np
import copy
from Transformation_Matrix import comp_matrix

# find what it takes to transform v1 to v0, return composed translation matrix

translation_offset = np.empty((2), dtype=float)											  # Needs to be upped to 3 for 3d
uniform_scale_factor = 0

def find_scale (pts):																	  # TODO: check for accuracy, close, but off in testing
	global translation_offset
	global uniform_scale_factor
	dist_0 = 0
	dist_1 = 0
	if (pts.shape == (2,2,3)):
		dist_0 = np.sqrt((pts[0,1,0]-pts[0,0,0])**2 + (pts[0,1,1]-pts[0,0,1])**2 + (pts[0,1,2]-pts[0,0,2])**2)	# Length of first simplex (3d)	for first vector: sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
		dist_1 = np.sqrt((pts[1,1,0]-pts[1,0,0])**2 + (pts[1,1,1]-pts[1,0,1])**2 + (pts[1,1,2]-pts[1,0,2])**2)	# Length of second simplex (3d)
		uniform_scale_factor = (dist_0/dist_1)
		translation_offset += np.append((uniform_scale_factor)*pts[0,0],0)
		return uniform_scale_factor*np.ones(3)
	else: 
		dist_0 = np.sqrt((pts[0,1,0]-pts[0,0,0])**2 + (pts[0,1,1]-pts[0,0,1])**2)		  # Length of first simplex (2d)	for first vector: sqrt((x2-x1)**2+(y2-y1)**2)
		dist_1 = np.sqrt((pts[1,1,0]-pts[1,0,0])**2 + (pts[1,1,1]-pts[1,0,1])**2)		  # Length of second simplex (2d)
		uniform_scale_factor = (dist_0/dist_1)
		translation_offset += ((uniform_scale_factor)*pts[0,0]) - pts[0,0]
		#print('target_pt', pts[0,0])
		print('uniform_scale_factor', dist_1/dist_0)
		print('scale_translation_offset_contribution',translation_offset)
		return uniform_scale_factor*np.ones(2)
	
def find_rot (pts):
	global translation_offset															  # access to the translation accumulator (still need to implement calc of offset)
	global uniform_scale_factor
	#temp_translation_offset = np.empty((2), dtype=float)
	v0 = pts[0,0] -	pts[0,1]															  # v0 = [x,y,z] = [i<hat>, j<hat>, k<hat>]
	v1 = pts[1,0] - pts[1,1]															  # v1 = [x,y,z] = [i<hat>, j<hat>, k<hat>]
	#print('rot_v', v0, v1)

	dx = pts[1,0,0] - pts[0,0,0]
	dy = pts[1,0,1] - pts[0,0,1]
	
	#print ('dx dy', dx, dy)
	#if (pts.shape == (2,2,3)):															  # get raw translation distances
	#	temp_translation_off = -(np.append(pts[1,0] - pts[0,0],0))
	#else:
	#	temp_translation_off = -(pts[1,0] - pts[0,0])


	alpha = np.arctan2(v1[1],v1[0]) - np.arctan2(v0[1],v0[0])							  # alpha angle about x-axis of standard basis (rad)
	r = np.sqrt(pts[0,0,0]**2 + pts[0,0,1]**2)											  # TODO: This needs to be calculate off of the point to be rotated BEFORE translation, not the alignment pt from S1
	alpha_prime = np.arctan2(v1[1],v1[0])												  # angle to alpha from x+
	s2_align_pt = np.array([((r) * np.cos(-alpha_prime)),
	 ((r) * np.sin(-alpha_prime))])														  # point as calculated from standard basis
	s2_align_pt *= uniform_scale_factor
	print ('target_alignment_pt', pts[0,0])
	#print ('alpha r', alpha, r)
	#print ('alpha_prime', alpha_prime)
	print ('s2_align_pt', s2_align_pt)
	#print ('usf',usf)
	#rotation_offset = s2_align_pt - pts[0,0]														  # failing due to rot_off sometimes being larger than the value of the pt it is being subtracted from
	#print ('rotation_offset', rotation_offset)
	#find quadrent & apply based on that for offset
	#if pts[1,0,0]>0 and pts[1,0,1]>0:													  # q1
	#	temp_offset *= np.array([1,-1])
	#	print('entered q1 rotation translation offest adjustment case')
	#elif pts[1,0,0]<0 and pts[1,0,1]>0:													  # q2
	#	temp_offset *= np.array([-1,-1])
	#	print('entered q2 rotation translation offest adjustment case')
	#elif pts[1,0,0]<0 and pts[1,0,1]<0:													  # q3
	#	temp_offset *= np.array([-1,1])
	#	print('entered q3 rotation translation offest adjustment case')
	#elif pts[1,0,0]>0 and pts[1,0,1]<0:													  # q4
	#	temp_offset *= np.array([1,1])													  # no change
	#	print('entered q4 rotation translation offest adjustment case')
	#elif pts[1,0,0]>0 and pts[1,0,1]==0:												  # x+
	#	temp_offset *= np.array([1,1])
	#	print('entered x+ rotation translation offest adjustment case')
	#elif pts[1,0,0]>0 and pts[1,0,1]==0:												  # x-
	#	temp_offset *= np.array([1,1])
	#	print('entered x- rotation translation offest adjustment case')
	#elif pts[1,0,0]==0 and pts[1,0,1]>0:												  # y+
	#	temp_offset *= np.array([-1,-1])
	#	print('entered y+ rotation translation offest adjustment case')
	#elif pts[1,0,0]==0 and pts[1,0,1]<0:												  # y-
	#	temp_offset *= np.array([-1,-1])
	#	print('entered y- rotation translation offest adjustment case')
	#elif pts[1,0,0]==0 and pts[1,0,1]==0:												  # orgin
	#	print ('WARNING: point at orgin in rotation offset, should only occur if point was at orgin initially (0,0,0) offset required')
	#print ('adj_temp_offset',temp_offset)
	print ('rotation_translation_offest_contribution', pts[0,0] - s2_align_pt)
	translation_offset += pts[0,0] - s2_align_pt 													  # check adjustment cases & application of tha adjustment (are we going the wrong way?)
	print ('combined scale && rotation translation_offset_contribution', translation_offset)

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
		#return -(np.append(pts[1,0] - pts[0,0],0)) #+ translation_offset 
	#return -(pts[1,0] - pts[0,0]) #+ translation_offset 
		return - translation_offset
	return - np.array([pts[1,0,0] - pts[0,0,0], pts[1,0,1] - pts[0,0,1]])# + translation_offset
	
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
	