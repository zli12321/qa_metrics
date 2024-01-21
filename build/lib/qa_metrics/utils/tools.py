import string
import contractions

def normalize_answer(text, lower=True):
    if isinstance(text, list):
        result = []
        for ele in text:
            ele = str(ele)
            if lower:
                ele = ele.lower()
            translator = str.maketrans('', '', string.punctuation)
            ele = ele.translate(translator)
            result.append(contractions.fix(' '.join(ele.split())))
        return result
    else:
        text = str(text)
        if lower:
            text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        return contractions.fix(' '.join(text.split()))