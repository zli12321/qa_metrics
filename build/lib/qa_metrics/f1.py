from .utils.tools import normalize_answer, remove_punctuation, lemmatize_text

'''
Return the F1 score, precision, and recall of the candidate answer given the reference answer
'''
def f1_score_with_precision_recall(reference, candidate):
    # Split the strings into sets of words
    reference = lemmatize_text(normalize_answer(str(reference)))
    candidate = lemmatize_text(normalize_answer(str(candidate)))
    words_reference = set(reference.split())
    words_candidate = set(candidate.split())

    # Calculate true positives, false positives, and false negatives
    tp = len(words_reference.intersection(words_candidate))
    fp = len(words_reference - words_candidate)
    fn = len(words_candidate - words_reference)

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'f1': f1_score, 'precision': precision, 'recall': recall}


'''
Return the F1 score of the candidate answer given the reference answer
'''
def f1_score(reference, candidate):
    f1_stats = f1_score_with_precision_recall(reference, candidate)
    return f1_stats['f1']



'''
Given a reference answer and a candidate answer, return True if the F1 score is greater than the threshold
'''
def f1_match(reference, candidate, threshold=0.5):
    if isinstance(reference, list) and isinstance(candidate, list):
        references = [lemmatize_text(normalize_answer(str(ele))) for ele in reference]
        candidates = [lemmatize_text(normalize_answer(str(ele))) for ele in candidate]

        f1_scores = []
        for reference in references:
            for candidate in candidates:
                f1_scores.append(f1_score(reference, candidate))

        return max(f1_scores) > threshold
    elif isinstance(reference, list):
        references = [lemmatize_text(normalize_answer(str(ele))) for ele in reference]
        candidate = lemmatize_text(normalize_answer(str(candidate)))

        f1_scores = []
        for reference in references:
            f1_scores.append(f1_score(reference, candidate))

        return max(f1_scores) > threshold
    elif isinstance(candidate, list):
        candidates = [lemmatize_text(normalize_answer(str(ele))) for ele in candidate]
        reference = lemmatize_text(normalize_answer(str(reference)))

        f1_scores = []
        for candidate in candidates:
            f1_scores.append(f1_score(reference, candidate))

        return max(f1_scores) > threshold
    else:
        reference = lemmatize_text(normalize_answer(str(reference)))
        candidate = lemmatize_text(normalize_answer(str(candidate)))

        return f1_score(reference, candidate) > threshold
    