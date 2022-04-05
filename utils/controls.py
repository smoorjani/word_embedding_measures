import numpy as np
from typing import List, Dict
from utils.data import get_data_property

def get_controls(data: List[Dict]) -> np.ndarray:
    """Creates controls for the regression coefficients.
    Note that you should add as many controls as possible.
    This is a relatively small set of controls meant for academic papers from the AMiner dataset.

    Args:
        data (List[Dict]): list of JSON objects containing all relevant data

    Returns:
        np.ndarray: controls stored in a numpy array
    """
    controls = []
    abstracts = get_data_property(data, "abstract")

    # additional controls
    venues = get_data_property(data, "venue")
    venue_to_idx = {}
    for venue in venues:
        if venue not in venue_to_idx:
            venue_to_idx[venue] = len(venue_to_idx)
    venue_indices = [venue_to_idx[venue] for venue in venues]
    controls.append(np.array(venue_indices))

    years = get_data_property(data, "year")
    controls.append(np.array(years))

    abstract_lens = [len(abstract) for abstract in abstracts]
    controls.append(np.array(abstract_lens))
    # gets the number of sentences in the abstract
    # split by . and removes items with less than 10 characters (for acronyms)
    num_sentences = [len(list(filter(lambda x: len(x) > 10, abstract.split('. ')))) for abstract in abstracts]
    controls.append(np.array(num_sentences))

    return np.stack(controls).transpose()
