import numpy as np
from scipy.stats.mstats import gmean
from utils.algs import two_opt, get_min_vol_ellipse

def get_features(chunk_emb: list, feature_list: list = ['speed', 'volume', 'circuitousness', 'distances'], volume_tolerance=0.01, circuitousness_tolerance=0.001) -> dict:
    """Wrapper function to get features

    Args:
        chunk_emb (list): list of chunk embeddings
        feature_list (list, optional): list of features to include. Defaults to ['speed', 'volume', 'circuitousness', 'distances'].
        volume_tolerance (float, optional): hyperparameter for volume. Defaults to 0.01.
        circuitousness_tolerance (float, optional): hyperparameter for the TSP problem for circuitousness. Defaults to 0.001.

    Returns:
        dict: returns a dict of features for the specified chunk embeddings
    """
    features_dict = {}
    distances = []
    for feature in feature_list:
        if feature.lower() == 'speed':
            distances, feature_val = get_speed(chunk_emb)

        elif feature.lower() == 'volume':
            feature_val = get_volume(chunk_emb, tolerance=volume_tolerance)

        elif feature.lower() == 'speed2': 
            feature_val = get_speed2(chunk_emb)

        elif feature.lower() == 'circuitousness':
            if not len(distances):
                distances, _ = get_speed(chunk_emb)
            feature_val = get_circuitousness(chunk_emb, tolerance=circuitousness_tolerance, distances=distances)

        elif feature.lower() == 'distances':
            if not len(distances):
                distances, _ = get_speed(chunk_emb)
            feature_val = distances

        features_dict[feature] = feature_val
    return features_dict

def get_speed(chunk_emb: list) -> tuple:
    """Gets the distances and speed of the chunk embeddings.

    Args:
        chunk_emb (list): list of chunk embeddings

    Returns:
        tuple[list, float]: returns the distances between each pair of elements in chunk emb and the average speed
    """
    chunk_emb = chunk_emb[~np.all(chunk_emb == 0, axis=1)]
    T = len(chunk_emb)
    distances = []
    for i in range(T - 1):
        distance = np.linalg.norm(chunk_emb[i+1] - chunk_emb[i])
        distances.append(distance)

    avg_speed = sum(distances) / (T-1)
    return distances, avg_speed
    
def get_speed2(chunk_emb: list, tolerance: float = 0.01, emb_dim: int = 300) -> float:
    """Gets the speed2 attribute of the chunk embeddings. See the appendix of the paper for more details.

    Args:
        chunk_emb (list): list of chunk embeddings

    Returns:
        float: returns the speed2 attribute of the chunk embeddings
    """
    chunk_emb = chunk_emb[~np.all(chunk_emb == 0, axis=1)]
    T = len(chunk_emb)
    speed2s = []
    logsumspeed2 = 0

    for i in range(T - 1):
        rank = np.linalg.matrix_rank(chunk_emb[:i+2], tolerance)
        if rank < emb_dim or (rank == emb_dim and chunk_emb[i].shape[0] <= emb_dim):
            tempA = chunk_emb[1:(i+2),:].transpose() - chunk_emb[0,:].transpose().reshape(-1, 1) @ np.ones((1,i + 1))
            U, S, _ = np.linalg.svd(tempA)
            S1 = U[:,:rank-1]
            tempP = np.vstack([(S1.transpose() @ tempA).transpose(), np.zeros((1, rank-1))])
            A, _ = get_min_vol_ellipse(tempP, 0.00001)
        else:
            A, _ = get_min_vol_ellipse(chunk_emb, 0.00001)

        U, S, _ = np.linalg.svd(A)
        D = -np.sum(np.log(np.sqrt(S)))
        speed2 = np.exp(D - logsumspeed2)
        logsumspeed2 = D
        speed2s.append(speed2)

    avg_speed2 = sum(speed2s) / (T-1)
    return avg_speed2

def get_volume(chunk_emb: list, tolerance: float = 0.01, emb_dim: int = 300) -> float:
    """Gets the volumne of the chunk embeddings.

    Args:
        chunk_emb (list): list of chunk embeddings
        tolerance (float, optional): tolerance for the Khachiyan algorithm. Defaults to 0.01.
        emb_dim (int, optional): dimension of word vectors. Defaults to 300.

    Returns:
        float: returns the volume of the chunk embeddings
    """
    P = chunk_emb

    rank = np.linalg.matrix_rank(P, tolerance)     
    if rank < emb_dim or (rank == emb_dim and P.shape[0] <= emb_dim):
        tempA = P[1:,:].transpose() - P[0,:].transpose().reshape(-1, 1) @ np.ones((1,P.shape[0] - 1))
        U, S, _ = np.linalg.svd(tempA)
        S1 = U[:,:rank-1]
        tempP = np.vstack([(S1.transpose() @ tempA).transpose(), np.zeros((1, rank-1))])
        A, _ = get_min_vol_ellipse(tempP)
    else:
        A, _ = get_min_vol_ellipse(P)

    U, S, _ = np.linalg.svd(A)
    return 1/gmean(np.sqrt(S))

def get_circuitousness(chunk_emb: list, tolerance: float = 0.001, distances: list = None) -> float:
    """Gets the circuitousness of the chunk embeddings.

    Args:
        chunk_emb (list): list of chunk embeddings
        tolerance (float, optional): tolerance for the TSP algorithm. Defaults to 0.001.
        distances (list, optional): distances between chunk_emb pairs. If not provided, it is calculated. Defaults to None.

    Returns:
        float: returns the circuitousness of the chunk embeddings
    """
    chunk_emb = chunk_emb[~np.all(chunk_emb == 0, axis=1)]
    T = len(chunk_emb)

    if not distances:
        distances, _ = get_speed(chunk_emb)
    
    distance_covered = sum(distances)

    if T > 2:
        # do travelling salesman problem
        route = two_opt(chunk_emb, tolerance)
        tsp = sum([np.linalg.norm(chunk_emb[route[i+1]] - chunk_emb[route[i]]) for i in range(len(route) - 1)])
    elif T == 2:
        tsp = distance_covered
    else:
        raise ValueError('You should have more than 2 chunks!')
    # ensure that minimum is not too low - skewing coefficient
    return np.log(distance_covered / tsp)
