import string
import contractions
import requests
import os

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

def download_link(file, url, name):
    if not os.path.isfile(file):
        print("Downloading {}...".format(name))
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(file, 'wb') as f:
                f.write(response.content)
            print("Download {} complete.".format(name))
        else:
            print("Failed to download the model. Status code:", response.status_code)