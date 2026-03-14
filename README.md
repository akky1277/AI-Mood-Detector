# 🧠 AI Mood Detector from Text
> B.Tech Final Year Project | NLP + Machine Learning | Python + Flask

---

## 📁 Project Structure

```
ai_mood_detector/
├── mood_engine.py      ← Core AI: dataset, preprocessing, model, prediction
├── app.py              ← Flask web server (2 routes: / and /predict)
├── run_cli.py          ← Run the terminal version (no browser needed)
├── requirements.txt    ← Python packages needed
├── model.pkl           ← Saved trained model (auto-created on first run)
├── vectorizer.pkl      ← Saved TF-IDF vectorizer (auto-created)
├── README.md           ← This file
└── templates/
    └── index.html      ← Beautiful web UI (dark theme, animated bars)
```

---

## ⚡ Quick Start

### Install dependencies
```bash
pip install flask scikit-learn pandas
```

### Option A – Web Interface (recommended)
```bash
python app.py
```
Then open → **http://127.0.0.1:5000**

### Option B – Terminal / CLI
```bash
python run_cli.py
```

---

## 🤖 How the AI Works (Step by Step)

```
User Input Text
      │
      ▼
┌─────────────────────┐
│   PREPROCESSING     │  lowercase + remove punctuation + remove stopwords
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   TF-IDF VECTORS    │  convert words → numbers (3000 features)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ LOGISTIC REGRESSION │  trained on 120 labeled sentences
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  PREDICTION +       │  emotion label + confidence % + key words
│  EXPLANATION        │
└─────────────────────┘
```

---

## 📊 Emotions Detected

| Emotion   | Emoji | Example Sentence                          |
|-----------|-------|-------------------------------------------|
| Happy     | 😊    | "I feel amazing today"                    |
| Sad       | 😢    | "I feel so lonely and empty inside"       |
| Angry     | 😠    | "I am so furious right now"               |
| Fear      | 😨    | "I am terrified of what might happen"     |
| Surprised | 😲    | "I cannot believe this just happened"     |
| Neutral   | 😐    | "I went to the market today"              |

---

## 🔬 Tech Stack

| Layer         | Technology                          |
|---------------|-------------------------------------|
| Language      | Python 3.x                          |
| ML Model      | Logistic Regression (scikit-learn)  |
| NLP/Features  | TF-IDF Vectorizer (1-gram + 2-gram) |
| Data          | pandas DataFrame                    |
| Web Framework | Flask                               |
| Frontend      | HTML + CSS + Vanilla JavaScript     |

---

## 🚀 Future Improvements

1. **Larger Dataset** – Load CSV with 10,000+ real labeled tweets/reviews
2. **Deep Learning** – Replace Logistic Regression with LSTM or BERT
3. **More Emotions** – Disgust, Contempt, Love, Excitement, Boredom
4. **Real-time Analysis** – Analyze as you type (debounced AJAX)
5. **History Log** – Save all predictions to SQLite database
6. **Export Report** – Download analysis as PDF
7. **Multi-language** – Support Hindi, Urdu, French etc.
8. **Voice Input** – Convert speech to text, then detect mood

---

## 🎤 Presentation Script (Speak This to Your Audience)

---

### Opening (30 seconds)
> "Hello everyone. Today I am going to present my project —
> the AI Mood Detector from Text.
> Have you ever wondered how apps like Google or Instagram 
> understand whether you're happy or angry just from your words?
> That is exactly what this project demonstrates."

---

### Problem Statement (20 seconds)
> "Humans express emotions through language every day —
> in messages, reviews, social media posts.
> But machines don't naturally understand feelings.
> This project solves that by teaching a machine to detect
> the emotion behind any sentence."

---

### How It Works (1 minute)
> "Let me walk you through the AI pipeline.
>
> First, the user types a sentence — for example:
> 'I feel amazing today.'
>
> Step one is Preprocessing. We clean the text —
> convert it to lowercase, remove punctuation,
> and remove common words like 'I', 'am', 'the'
> which carry no emotional meaning.
>
> Step two is Feature Extraction using TF-IDF.
> This converts each word into a number so the machine
> can process it mathematically.
> TF-IDF gives higher scores to words that are rare
> but meaningful — like 'furious' or 'heartbroken'.
>
> Step three is the Machine Learning model —
> Logistic Regression. It was trained on 120 sentences
> labeled with 6 emotions. It learned patterns like:
> 'amazing' and 'excited' → Happy.
> 'furious' and 'livid' → Angry.
>
> Finally, the model outputs the predicted emotion
> with a confidence percentage and highlights
> which specific words caused that prediction."

---

### Live Demo (1 minute)
> "Let me show you a live demo.
> [Type: 'I am so furious right now']
> The AI predicts: Angry — with high confidence.
> The key words it detected are: furious.
>
> Now watch what happens with a different sentence.
> [Type: 'I feel heartbroken and empty inside']
> Predicted: Sad — and the key words are: heartbroken, empty.
>
> Even this neutral sentence:
> [Type: 'I went to the market today']
> Shows Neutral — because there are no emotional words."

---

### Technical Highlights (30 seconds)
> "Key technical decisions I made:
> I used TF-IDF with bigrams — that means the model
> also considers 2-word phrases like 'feel amazing'
> which carry more context than single words.
> I chose Logistic Regression because it is fast,
> interpretable, and gives probability scores
> for all classes — not just a single answer."

---

### Closing (20 seconds)
> "To summarize: this project demonstrates the complete
> NLP pipeline — from raw text to emotion prediction —
> using real machine learning techniques.
> Future extensions include using larger datasets,
> deep learning models like BERT, and adding voice input.
> Thank you. I am happy to take any questions."

---
