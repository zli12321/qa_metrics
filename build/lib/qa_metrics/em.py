import string
from .utils.tools import normalize_answer, remove_punctuation

def em_match(reference, candidate):
    if isinstance(reference, list) and isinstance(candidate, list):
        reference = [remove_punctuation(normalize_answer(str(ele))) for ele in reference]
        candidate = [remove_punctuation(normalize_answer(str(ele))) for ele in candidate]
    elif isinstance(reference, list):
        reference = [remove_punctuation(normalize_answer(str(ele))) for ele in reference]
        candidate = [remove_punctuation(normalize_answer(str(candidate)))]
    elif isinstance(candidate, list):
        candidate = [remove_punctuation(normalize_answer(str(ele))) for ele in candidate]
        reference = [remove_punctuation(normalize_answer(str(reference)))]
    else:
        '''
        Normalize the strings
        '''
        reference = [remove_punctuation(normalize_answer(str(reference)))]
        candidate = [remove_punctuation(normalize_answer(str(candidate)))]

    for ref in reference:
        for can in candidate:
            if ref in can:
                return True
            
    return False