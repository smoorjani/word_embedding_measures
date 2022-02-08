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

def get_volume(chunks_emb: list):
    # TODO: implement this
    pass

def get_circuitousness(chunks_emb: list):
    # TODO: implement this
    pass