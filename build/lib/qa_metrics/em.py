import string
from .utils.tools import normalize_answer

def em_match(reference, candidate):
    if isinstance(reference, list) and isinstance(candidate, list):
        reference = [normalize_answer(str(ele)) for ele in reference]
        candidate = [normalize_answer(str(ele)) for ele in candidate]

        return bool(set(reference) & set(candidate))
    elif isinstance(reference, list):
        reference = [normalize_answer(str(ele)) for ele in reference]
        candidate = normalize_answer(str(candidate))
        return bool(set(reference) & set([candidate]))
    elif isinstance(candidate, list):
        candidate = [normalize_answer(str(ele)) for ele in candidate]
        reference = normalize_answer(str(reference))
        return bool(set([reference]) & set(candidate))
    else:
        '''
        Normalize the strings
        '''
        reference = normalize_answer(str(reference))
        candidate = normalize_answer(str(candidate))

        return reference == candidate