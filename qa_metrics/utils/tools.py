import string
import contractions
import requests
import os
from datetime import datetime

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

def file_needs_update(url, file_path):
    """
    Check if the file at the given path needs to be updated based on the
    Last-Modified header from the file URL.
    """
    try:
        response = requests.head(url)
        if response.status_code == 200 and 'Last-Modified' in response.headers:
            remote_last_modified = requests.utils.parsedate_to_datetime(response.headers['Last-Modified'])
            if not os.path.exists(file_path):
                return True  # File does not exist, needs download.
            local_last_modified = datetime.fromtimestamp(os.path.getmtime(file_path), tz=remote_last_modified.tzinfo)
            return remote_last_modified > local_last_modified
    except requests.RequestException as e:
        print(f"Error checking if file needs update: {e}")
    return False  # Default to not updating if we can't determine.