import numpy as np
from scipy.spatial.transform import Rotation


def align_rvec(ear2ear: np.ndarray, chin2head: np.ndarray):
    assert ear2ear.shape == chin2head.shape == (3,)
    ear2ear = ear2ear / np.linalg.norm(ear2ear)
    chin2head = chin2head / np.linalg.norm(chin2head)

    assert abs(ear2ear @ chin2head) < .01, \
        f'vectors must be orthogonal, got {ear2ear @ chin2head}'

    x, *_ = chin2head
    _, y, z = ear2ear

    return -np.array([
        np.arccos(x) - np.pi/2,
        np.arccos(y),
        np.arccos(z)])


if __name__ == '__main__':
    ear2ear = np.array([0, 0, 1])
    chin2head = np.array([0, 1, 0])
    r = align_rvec(ear2ear, chin2head)

    rmat = Rotation.from_rotvec(r).as_matrix()

    print(rmat @ ear2ear)
    print(rmat @ chin2head)
