import numpy as np

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

def get_volume(chunks_emb: list, tol: float):
    # Code based from
    # Nima Moshtagh (nima@seas.upenn.edu)
    # University of Pennsylvania
    d = len(chunks_emb)
    N = len(chunks_emb[0])
    Q = np.zeros(d+1,N)
    Q[1:d,:] = chunks_emb[1:d,1:N]
    Q[d+1,:] = np.ones[1,N]
    count = 1
    err = 1
    u = (1/N) * np.ones(N,1)          # 1st iteration

    # Khachiyan Algorithm
    while err > tol:
        X = Q * np.diag(u) * np.transpose(Q)       # X = \sum_i ( u_i * q_i * q_i')  is a (d+1)x(d+1) matrix
        M = np.diag(np.transpose(Q) * np.inv(X) * Q)  #M the diagonal vector of an NxN matrix
        j = max(M)
        step_size = (j - d -1)/((d+1)*(j-1))
        new_u = (1 - step_size)*u
        new_u[j] = new_u[j] + step_size
        count = count + 1
        err = np.norm(new_u - u)
        u = new_u
    
    ''' Computing the Ellipse parameters
     Finds the ellipse equation in the 'center form': 
     (x-c)' * A * (x-c) = 1
     It computes a dxd matrix 'A' and a d dimensional vector 'c' as the center
     of the ellipse. 
    '''
    U = np.diag(u)

    # the A matrix for the ellipse
    A = (1/d) * np.inv(chunks_emb * U * np.transpose(chunks_emb) - (chunks_emb * u)*np.transpose(chunks_emb*u) )

    # center of the ellipse 
    c = chunks_emb * u
    pass

def get_circuitousness(chunks_emb: list):
    """[summary]

    Args:
        chunks_emb (list): [description]

    Returns:
        [type]: [description]
    """
    distances, avg_speed = get_speed(chunks_emb)
    minimum = np.min(distances)
    circuitousness = sum(distances)/minimum
    return circuitousness