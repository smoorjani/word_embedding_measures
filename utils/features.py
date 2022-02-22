import numpy as np
from scipy.stats.mstats import gmean

def get_speed(chunks_emb: list):
    """[summary]

    Args:
        chunks_emb (list): [description]

    Returns:
        [type]: [description]
    """
    T = len(chunks_emb)
    distances = []
    for i in range(T - 1):
        distance = np.linalg.norm(chunks_emb[i-1] - chunks_emb[i])
        distances.append(distance)

    avg_speed = sum(distances) / (T-1)
    return distances, avg_speed

    
def get_volume(chunk_emb, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
            [x,y,z,...],
            [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    # P = np.stack(chunk_emb, axis=0)
    P = chunk_emb
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(np.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = np.linalg.inv(
                    np.dot(P.T, np.dot(np.diag(u), P)) - 
                    np.array([[a * b for b in center] for a in center])
                    ) / d
                    
    # Get the values we'd like to return
    try:
        U, s, rotation = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        print(chunk_emb)
        # TODO: fix whatever is going wrong here
        return 0

    return getEllipsoidVolume(np.sqrt(s))/gmean(np.sqrt(np.diagonal(s)))
    # radii = np.sqrt(s)
    
    # return (center, radii, rotation)
    # return getEllipsoidVolume(radii)

def getEllipsoidVolume(radii):
    """Calculate the volume of the blob"""
    return 4./3.*np.pi*radii[0]*radii[1]*radii[2]

def get_circuitousness(chunks_emb: list, eps: float = 0.05):
    """[summary]

    Args:
        chunks_emb (list): [description]

    Returns:
        [type]: [description]
    """
    distances, avg_speed = get_speed(chunks_emb)

    # ensure that minimum is not too low - skewing coefficient
    minimum = 0
    if np.max(distances) < eps:
        minimum = eps
    else:
        minimum = np.min([d for d in distances if d > eps])
    circuitousness = sum(distances)/minimum

    if circuitousness > 1000:
        print(distances, minimum, circuitousness)
    return circuitousness