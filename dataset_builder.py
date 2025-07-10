from bs4 import BeautifulSoup
import requests

#vraca poziciju slova s naglaskom (pozicija>=1 ako u rijeci ima naglasak, pozicija = 0 kad je nenaglasena rijec)

def is_palatal(letter):
    palatal_letters = "čćđšž"
    return letter in palatal_letters

def idx(word):
    indexes = [idx+1 for (idx, letter) in enumerate(word) if (not(letter.isascii()) and not(is_palatal(letter)))]
    if indexes==[]:
        return 0
    else:
        return indexes[0]
    


#dodavanje parova (rijec s naglascima, pozicija naglaska) u bazu rijeci
def build_dataset():
    letters = "abcdefghijklmnoprstuvz"
    url = 'https://rjecnik.hr/'
    dataset = {}

    for letter in letters:
        for k in range(1,5):
            payload = {'letter':letter, 'page':k}
            result = requests.get(url, params=payload)

            soup = BeautifulSoup(result.content, features="html.parser")
            entries = soup.find_all("span", attrs={"class": "word"})
            for s in entries:
                title = s.text.strip().partition(' ')[0]
                dataset[title] = idx(title)

    return dataset





