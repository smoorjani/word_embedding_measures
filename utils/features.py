import numpy as np
from scipy.stats.mstats import gmean

def get_speed(chunks_emb: list):
    """[summary]

    Args:
        chunks_emb (list): [description]

    Returns:
        [type]: [description]
    """
    chunks_emb = chunks_emb[~np.all(chunks_emb == 0, axis=1)]
    T = len(chunks_emb)
    distances = []
    for i in range(T - 1):
        distance = np.linalg.norm(chunks_emb[i+1] - chunks_emb[i])
        distances.append(distance)

    avg_speed = sum(distances) / (T-1)
    return distances, avg_speed

def get_min_vol_ellipse(P, tolerance=0.01):
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
    return A, center
    
def get_volume(chunk_emb, tolerance=0.01, emb_dim=300):
    P = chunk_emb

    rank = np.linalg.matrix_rank(P, tolerance)     
    if rank < emb_dim or (rank == emb_dim and P.shape[0] <= emb_dim):
        print(P.shape, P[1:,:].transpose().shape, P[0,:].transpose().shape, np.ones((1,P.shape[0] - 1)).shape)
        tempA = P[1:,:].transpose() - P[0,:].transpose().reshape(-1, 1) @ np.ones((1,P.shape[0] - 1))
        U, S, V = np.linalg.svd(tempA)
        S1 = U[:,1:rank]
        print(rank, U.shape, S1.shape, tempA.shape)
        print((S1.transpose() @ tempA).shape, np.zeros((1, rank-1)).shape)
        tempP = np.vstack([(S1.transpose() @ tempA).transpose(), np.zeros((1, rank-1))])
        A, center = get_min_vol_ellipse(tempP)
    else:
        A, center = get_min_vol_ellipse(P)

        # [U,S,V]=svd(shapes(i).position_window(2:end,:)'-shapes(i).position_window(1,:)'*ones(1,shapes(i).nwindows-1));
            # S1=U(:,1:ndim-1);
            # [A,c]=MinVolEllipse([S1'*(shapes(i).position_window(2:end,:)'-shapes(i).position_window(1,:)'*ones(1,shapes(i).nwindows-1)) zeros(ndim-1,1)],0.00001);
            
    # Get the values we'd like to return
    # try:
    #     U, S, V = np.linalg.svd(A)
    # except np.linalg.LinAlgError:
    #     print(chunk_emb)
    #     # TODO: fix whatever is going wrong here
    #     return 0
    U, S, V = np.linalg.svd(A)
    print(S)
    return 1/gmean(np.sqrt(S))
    # radii = np.sqrt(s)
    
    # return (center, radii, V)
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
    chunks_emb = chunks_emb[~np.all(chunks_emb == 0, axis=1)]
    T = len(chunks_emb)

    distances, _ = get_speed(chunks_emb)
    distance_covered = sum(distances)

    if T > 2:
        # do travelling salesman problem
        route = two_opt(chunks_emb,0.001)
        tsp = sum([np.linalg.norm(chunks_emb[route[i+1]] - chunks_emb[route[i]]) for i in range(len(route) - 1)])
    elif T == 2:
        tsp = distance_covered
    else:
        raise ValueError('You should have more than 2 chunks!')
    # ensure that minimum is not too low - skewing coefficient
    return distance_covered / tsp

## Travelling Salesman Problem
## Code taken from https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy
# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(cities,improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
    return route # When the route is no longer improving substantially, stop searching and return the route.