import numpy as np

def comp_matrix(scale, rotation, shear, translation):
	# should filter inputs if accepting error prone input (dtype, length, and domains)
	Tx = translation[0]
	Ty = translation[1]										# eff: we only ever use these variables once, no need to call them, parse input as needed
	Tz = 0 if translation.size < 3 else translation[2]
	Sx = scale[0]
	Sy = scale[1]
	Sz = 1 if scale.size < 3 else scale[2]
	Shx = shear[0]
	Shy = shear[1]
	Shz = 0 if shear.size < 3 else shear[2]
	Rxc, Rxs = np.cos(rotation[0]), np.sin(rotation[0])
	Ryc, Rys = np.cos(rotation[1]), np.sin(rotation[1])		# eff: we call these variables multiple times, create standalones for efficency   if Shx else 0
	Rzc, Rzs = (1,0) if rotation.size < 3 else (np.cos(rotation[2]), np.sin(rotation[2]))

	T_M = np.array([[1, 0, 0, Tx],
                    [0, 1, 0, Ty],
                    [0, 0, 1, Tz],
                    [0, 0, 0, 1]])
	S_M = np.array([[Sx, 0, 0, 0],
                    [0, Sy, 0, 0],
                    [0, 0, Sz, 0],
                    [0, 0, 0, 1]])
	Sh_M = np.array([[1, Shy/Shx, Shz/Shx, 0],
                     [Shx/Shy, 1, Shz/Shy, 0],
                     [Shx/Shz, Shy/Shz, 1, 0],
                     [0, 0, 0, 1]])
	Rx_M = np.array([[1, 0, 0, 0],
                     [0, Rxc, -Rxs, 0],
                     [0, Rxs, Rxc, 0],
                     [0, 0, 0, 1]])
	Ry_M = np.array([[Ryc, 0, Rys, 0],
                     [0, 1, 0, 0],
                     [-Rys, 0, Ryc, 0],
                     [0, 0, 0, 1]])
	Rz_M = np.array([[Rzc, -Rzs, 0, 0],
                     [Rzs, Rzc, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

	return np.dot(S_M,np.dot(T_M,np.dot(Rz_M,np.dot(Ry_M,Rx_M))))						  # IMPORTANT: the transformations must be multiplied together in the [B]reverse order[/B] to that in which we want them applied

def decomp_matrix(transformation_matrix):
	tm = transformation_matrix
	translation = np.array([tm[0,3], tm[1,3], tm[2,3]])
	scale = np.array([np.abs(np.sqrt(np.pow(tm[0,0],2)+np.pow(tm[1,0],2)+np.pow(tm[2,0],2))),
					  np.abs(np.sqrt(np.pow(tm[0,1],2)+np.pow(tm[1,1],2)+np.pow(tm[2,1],2))),
					  np.abs(np.sqrt(np.pow(tm[0,2],2)+np.pow(tm[1,2],2)+np.pow(tm[2,2],2)))])
	rotation = np.array([np.arctan2(tm[2,1]/scale[1],tm[2,2],scale[2]),
						 np.arctan2(-tm[2,0]/scale[0],np.sqrt(np.pow(tm[2,1]/scale[1],2)+np.pow(tm[2,2]/scale[2],2))),
						 np.arctan2(tm[1,0]/scale[0],tm[0,0]/scale[0])])
	shear = np.zeros(4)										# need support for shear

	return 	np.array(translation, scale, rotation, shear)
