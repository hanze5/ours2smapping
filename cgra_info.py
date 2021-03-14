import numpy as np

pea_4x4_routes = np.array(
# line 0
[0,1,0,0,0, 1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
 1,0,1,0,0, 0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
 0,1,0,1,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
 0,0,1,0,1, 0,0,0,1,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
 0,0,0,1,0, 0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,

 1,0,0,0,0, 0,1,0,0,0, 1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
 0,1,0,0,0, 1,0,1,0,0, 0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
 0,0,1,0,0, 0,1,0,1,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0,
 0,0,0,1,0, 0,0,1,0,1, 0,0,0,1,0, 0,0,0,0,0, 0,0,0,0,0,
 0,0,0,0,1, 0,0,0,1,0, 0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0,

 0,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 1,0,0,0,0, 0,0,0,0,0,
 0,0,0,0,0, 0,1,0,0,0, 1,0,1,0,0, 0,1,0,0,0, 0,0,0,0,0,
 0,0,0,0,0, 0,0,1,0,0, 0,1,0,1,0, 0,0,1,0,0, 0,0,0,0,0,
 0,0,0,0,0, 0,0,0,1,0, 0,0,1,0,1, 0,0,0,1,0, 0,0,0,0,0,
 0,0,0,0,0, 0,0,0,0,1, 0,0,0,1,0, 0,0,0,0,1, 0,0,0,0,0,

 0,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 1,0,0,0,0,
 0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,0, 1,0,1,0,0, 0,1,0,0,0,
 0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,1,0,1,0, 0,0,1,0,0,
 0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,0, 0,0,1,0,1, 0,0,0,1,0,
 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1, 0,0,0,1,0, 0,0,0,0,1,

 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0,
 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,0, 1,0,1,0,0,
 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,1,0,1,0,
 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,0, 0,0,1,0,1,
 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1, 0,0,0,1,0
 ]

)