from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pickle
import re
import string

app = FastAPI()

# Load LDA components
lda_model = LdaModel.load("lda_model.gensim")
dictionary = Dictionary.load("dictionary.gensim")

# Topic labels (customize if needed)
TOPIC_LABELS = {
    0: "Auto",
    1: "Tech",
    2: "Sports",
    3: "Politics",
    4: "International"
}

# Load Marathi stopwords
def load_stopwords(file_path='marathi_stopwords.txt'):
    with open(file_path, 'r', encoding='utf-8') as file:
        return {line.strip() for line in file if line.strip()}

stopwords = load_stopwords()

# Cleaning and preprocessing
def clean_text(text):
    all_punct = string.punctuation + "।"
    text = re.sub(f"[{re.escape(all_punct)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 2]
    return " ".join(tokens)

def remove_stopwords(text):
    tokens = text.split()
    return ' '.join([token for token in tokens if token not in stopwords])

def preprocess_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Keep only Marathi chars
    text = clean_text(text)
    text = remove_stopwords(text)
    return text.split()

# Topic prediction
def get_topic(text):
    tokens = preprocess_text(text)
    bow = dictionary.doc2bow(tokens)
    topic_dist = lda_model.get_document_topics(bow; minimum_probability=0.0)
    labeled_topics = []
    for topic_id, prob in topic_dist:
        label = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")
        labeled_topics.append((label, prob))

    # Sort by probability (descending)
    labeled_topics.sort(key=lambda x: -x[1])
    return labeled_topicss

# HTML form
@app.get("/", response_class=HTMLResponse)
async def form_get():
    return """
    <html>
    <head>
        <title>Marathi News Topic Identifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            textarea { width: 100%; height: 150px; font-size: 1em; }
            input[type=submit] { padding: 10px 20px; font-size: 1em; }
        </style>
    </head>
    <body>
        <h2>Marathi News Topic Identifier</h2>
        <form action="/predict" method="post">
            <textarea name="text" placeholder="Type or paste Marathi news here..."></textarea><br><br>
            <input type="submit" value="Identify Topic">
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def form_post(text: str = Form(...)):
    topics = get_topic(text)
    top_topic = topics[0][0] if topics else "Unknown"

    result_html = f"""
    <html>
    <head><title>Result</title></head>
    <body>
        <h2>Predicted Topic: {top_topic}</h2>
        <h3>Sorted Topic Probabilities:</h3>
        <ul>
    """
    for label, prob in topics:
        result_html += f"<li><b>{label}</b>: {prob*100:.2f}%</li>"
    result_html += """
        </ul><br><a href='/'>Try Another</a>
    </body>
    </html>
    """
    return result_html
