# Marathi News Topic Modeling

This repository contains experiments for topic modeling on Marathi news articles using the L3Cube Marathi LDC dataset. The project explores both **Latent Semantic Analysis (LSA)** and **Latent Dirichlet Allocation (LDA)** (via Gensim) to uncover latent topics in the dataset. Additionally, a custom Marathi stopwords list has been created and utilized during text preprocessing.

---

## Overview

- **Dataset:**  
  The analysis is performed on the [L3Cube Marathi LDC dataset](https://github.com/l3cube-pune/MarathiNLP/tree/main/L3Cube-MahaNews/LDC). This dataset consists of Marathi news articles collected from various sources.

- **Methods Explored:**  
  - **Latent Semantic Analysis (LSA):**  
    An exploratory approach to extract topics from the news articles.
  - **Latent Dirichlet Allocation (LDA):**  
    Implemented using the Gensim library, this method is used for robust topic extraction.

- **Custom Stopwords:**  
  A curated stopwords list tailored to Marathi news text is used to filter out non-informative words during preprocessing.

---

## Web Application

Two variants of the topic identification web app are provided in the repository:

- `main.py` – a feature-rich version with intelligent topic inference and a pie chart visualization  
- `app-1-demo.py` – a simplified or demo variant for lightweight testing

## Output
The Output/ folder contains visual examples of the web app’s predictions.
Input `(1 Label)`:
https://github.com/Pranav1173/NLP-Marathi-News-Topic-Identifier/blob/main/Output/Web-Input-1Label.jpg
### To Run the Web App:

```bash
uvicorn main:app --reload

