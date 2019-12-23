# Point-Pattern-Matching-Convex-Hull
Implementation of the following: https://ieeexplore.ieee.org/abstract/document/6313439
- Point Patern Matching (PPM) of two point sets via taking the convex hull of those pointsets as a represent subset. 
- Any differences between the two sets of points are statistically more likely to occur in the middle vs. the edges.
- One set of points is transformed to attempt to match the other, matches and RMSE is logged
- The transformation matrix for the best match is then applied
- Backend: Transformation matricies are determined by vector comparisons & applied

Todo: - Shear support
      - Non-uniform scaling support
      - Cython optimizations/ parellel processing optimizations
      - Take another stab at the 3D beastie
