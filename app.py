from flask import Flask, request, render_template_string, url_for
from flask_assets import Environment, Bundle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import re

MODEL_PATH = 'model.pt'
DATA_CSV = 'unaccented_data.csv'
MAX_WORD_LEN = 16
ALLOWED_RE = re.compile(r'^[A-Za-zčćđšžČĆĐŠŽ-]{1,16}$')

df = pd.read_csv(DATA_CSV)
words = df['Word'].astype(str).tolist()
chars = sorted({c for w in words for c in w})
char2idx = {c: i+1 for i, c in enumerate(chars)}
vocab_size = len(char2idx) + 1
max_len = max(len(w) for w in words)

class AccentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.dropout(h_final)
        return self.fc(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AccentModel(vocab_size, embed_dim=64, hidden_dim=64,
                    output_dim=max_len+1, dropout_p=0.3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

app = Flask(__name__)

assets = Environment(app)

assets.load_path = [
    'static/scss'
]

scss_bundle = Bundle(
    'palette.scss',
    'style.scss',
    filters='pyscss',
    output='css/main.css'
)

assets.register('main_css', scss_bundle)

app.config['ASSETS_DEBUG'] = True   # disable caching so you see CSS changes immediately
assets.auto_build = True




HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediktor pozicije naglaska</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
</head>
<body>
  <h1>Prediktor pozicije naglaska u riječi</h1>
  <form method="post">
    <label for="word">Upiši riječ:</label><br>
    <input type="text" id="word" name="word" value="{{word|default('')}}" autofocus>
    <button type="submit">Pronađi naglasak</button>
  </form>
  {% if error %}<p class="error">{{error}}</p>{% endif %}
  {% if results %}
    <div class="result">
      {% for part, preds in results %}
        <h3>Riječ: '{{part}}'</h3>
        <ol>
        {% for pos, char, prob in preds %}
          <li>
            {% if pos==0 %}
              Nenaglašena riječ ({{prob}}%)
            {% else %}
              Naglasak je na slovu '{{char}}' na poziciji {{pos}} ({{prob}}%)
            {% endif %}
          </li>
        {% endfor %}
        </ol>
      {% endfor %}
    </div>
  {% endif %}
</body>
</html>
'''

def predict_top2(segment: str):
    seq = torch.tensor([char2idx.get(c, 0) for c in segment], dtype=torch.long)
    length = torch.tensor([len(seq)], dtype=torch.long)
    padded = pad_sequence([seq], batch_first=True, padding_value=0).to(device)
    length = length.to(device)
    with torch.no_grad():
        logits = model(padded, length).squeeze(0)
        probs = F.softmax(logits, dim=0)
        top2 = torch.topk(probs, 2)
    results = []
    for idx, p in zip(top2.indices.tolist(), top2.values.tolist()):
        pos = idx  # 0=no accent, 1-based positions otherwise
        char = segment[pos-1] if 1 <= pos <= len(segment) else None
        if char is not None and char in "aeiouAEIOUrR":
            results.append((pos, char, round(p*100, 2)))
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    results = None
    word = ''
    if request.method == 'POST':
        word = request.form.get('word', '').strip()
        if not ALLOWED_RE.match(word):
            error = 'Riječ može imati najviše 16 slova i sadržavati samo slova hrvatske abecede.'
        else:
            parts = word.split('-')
            results = [(part, predict_top2(part)) for part in parts]
    return render_template_string(HTML, error=error,
                                  results=results, word=word)

if __name__ == '__main__':
    app.run(debug=True)
