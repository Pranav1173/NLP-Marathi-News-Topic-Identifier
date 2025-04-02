from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import plotly.graph_objects as go
import re
import string
import uuid

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

# Intelligent label inference based on probability distribution
def infer_labels_from_distribution(sorted_topics):
    if not sorted_topics:
        return ["Uncertain"]

    probs = [prob for _, prob in sorted_topics]

    if probs[0] >= 0.65:
        return [sorted_topics[0][0]]
    elif len(probs) >= 2 and (probs[0] + probs[1]) >= 0.80:
        return [sorted_topics[0][0], sorted_topics[1][0]]
    elif len(probs) >= 3 and (probs[0] + probs[1] + probs[2]) >= 0.75:
        return [sorted_topics[0][0], sorted_topics[1][0], sorted_topics[2][0]]
    else:
        return [sorted_topics[0][0]]

# Topic prediction
def get_topic(text):
    tokens = preprocess_text(text)
    bow = dictionary.doc2bow(tokens)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)

    labeled_topics = []
    for topic_id, prob in topic_dist:
        label = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")
        labeled_topics.append((label, prob))

    labeled_topics.sort(key=lambda x: -x[1])
    return labeled_topics

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
    inferred_labels = infer_labels_from_distribution(topics)
    label_output = ", ".join(inferred_labels)

    # Prepare data for pie chart
    labels = [label for label, _ in topics]
    values = [prob for _, prob in topics]
    chart_id = f"chart_{uuid.uuid4().hex}"

    pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    pie_html = pie_chart.to_html(full_html=False, include_plotlyjs='cdn', div_id=chart_id)

    result_html = f"""
    <html>
    <head>
        <title>Result</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .topic-box {{
                background-color: #001f3f;
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-size: 1.2em;
                margin-bottom: 20px;
            }}
            .center {{
                text-align: center;
            }}
            .try-again {{
                margin-top: 30px;
                display: inline-block;
                padding: 10px 20px;
                background-color: #0074D9;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 1em;
                text-decoration: none;
            }}
            .try-again:hover {{
                background-color: #005fa3;
            }}
        </style>
    </head>
    <body>
        <div class="topic-box">
            <strong>Predicted Topics:</strong> {label_output}
        </div>

        <h3>Sorted Topic Probabilities:</h3>
        <ul>
    """
    for label, prob in topics:
        result_html += f"<li><b>{label}</b>: {prob*100:.2f}%</li>"

    result_html += f"""
        </ul>

        <h3>Topic Distribution (Pie Chart):</h3>
        {pie_html}

        <div class="center">
            <a href="/" class="try-again">🔁 Try Another</a>
        </div>
    </body>
    </html>
    """

    return result_html
