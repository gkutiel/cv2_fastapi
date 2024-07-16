import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == '__main__':
    ear2ear = np.array([0, 0, 1])
    chin2head = np.array([0, 1, 0])
    r = align_rvec(ear2ear, chin2head)

    rmat = Rotation.from_rotvec(r).as_matrix()

    print(rmat @ ear2ear)
    print(rmat @ chin2head)
