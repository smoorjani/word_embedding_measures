import numpy as np
from utils.data import get_data_property

def get_controls(data):
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
