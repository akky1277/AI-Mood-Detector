"""
================================================================
  AI MOOD DETECTOR - Core Engine
  File: mood_engine.py

  This file contains EVERYTHING the AI needs to work:
    1. Dataset (sentences + emotion labels)
    2. Text Preprocessing (cleaning the text)
    3. Feature Extraction (TF-IDF)
    4. Model Training (Logistic Regression)
    5. Prediction + Explanation

  Think of this file as the "brain" of our AI project.
================================================================
"""

import re                              # For removing punctuation using patterns
import string                          # For the list of punctuation characters
import pandas as pd                    # For creating and managing the dataset table
from sklearn.feature_extraction.text import TfidfVectorizer   # Converts text → numbers
from sklearn.linear_model import LogisticRegression            # Our ML model
from sklearn.model_selection import train_test_split           # Split data for testing
from sklearn.metrics import accuracy_score, classification_report
import pickle                          # For saving/loading the trained model
import os


# ================================================================
#  STEP 1: DATASET
#  We create our own labeled dataset.
#  Each sentence is paired with an emotion label.
#  In a real project you'd load a CSV with thousands of rows —
#  but this hand-crafted set is perfect for learning.
# ================================================================

DATASET = [

    # ── HAPPY ────────────────────────────────────────────────────
    ("I feel amazing today",                        "Happy"),
    ("This is the best day of my life",             "Happy"),
    ("I am so excited about this",                  "Happy"),
    ("Everything is going perfectly well",          "Happy"),
    ("I love spending time with my family",         "Happy"),
    ("I just got promoted at work, I am thrilled",  "Happy"),
    ("We won the match, I am overjoyed",            "Happy"),
    ("She smiled and made my whole day better",     "Happy"),
    ("I am grateful for all the good things",       "Happy"),
    ("Life feels wonderful and full of joy",        "Happy"),
    ("I passed my exam with excellent marks",       "Happy"),
    ("Today was a fantastic and beautiful day",     "Happy"),
    ("I feel cheerful and full of energy",          "Happy"),
    ("My friends surprised me with a party",        "Happy"),
    ("I am delighted to meet you here",             "Happy"),
    ("This good news made me jump with joy",        "Happy"),
    ("I feel blessed and very lucky",               "Happy"),
    ("Laughing with friends makes me so happy",     "Happy"),
    ("I am on top of the world right now",          "Happy"),
    ("Everything is sunshine and rainbows today",   "Happy"),

    # ── SAD ──────────────────────────────────────────────────────
    ("I feel so lonely and empty inside",           "Sad"),
    ("Nobody understands what I am going through",  "Sad"),
    ("I lost my job and I feel hopeless",           "Sad"),
    ("I am heartbroken after what happened",        "Sad"),
    ("Everything feels dark and meaningless",       "Sad"),
    ("I miss my loved ones so much",                "Sad"),
    ("I cried myself to sleep last night",          "Sad"),
    ("I feel like giving up on everything",         "Sad"),
    ("Life is so unfair and painful",               "Sad"),
    ("I am deeply disappointed in myself",          "Sad"),
    ("I have been feeling low and depressed lately","Sad"),
    ("No one cares about how I feel",               "Sad"),
    ("I feel broken and completely lost",           "Sad"),
    ("Today was a terrible and sad day for me",     "Sad"),
    ("My heart aches thinking about the past",      "Sad"),
    ("I failed again and I feel worthless",         "Sad"),
    ("I am grieving and cannot stop crying",        "Sad"),
    ("The sadness is overwhelming me",              "Sad"),
    ("I feel abandoned by everyone I trusted",      "Sad"),
    ("Nothing brings me joy anymore",               "Sad"),

    # ── ANGRY ────────────────────────────────────────────────────
    ("I am so furious right now",                   "Angry"),
    ("This makes me absolutely livid",              "Angry"),
    ("I cannot believe how unfair this is",         "Angry"),
    ("Stop lying to me, I am done with this",       "Angry"),
    ("I am extremely frustrated with everything",   "Angry"),
    ("How dare they treat me this way",             "Angry"),
    ("I am boiling with rage inside",               "Angry"),
    ("This is completely unacceptable behavior",    "Angry"),
    ("I hate being treated like I do not matter",   "Angry"),
    ("They broke my trust and I am outraged",       "Angry"),
    ("I am fed up with all these lies",             "Angry"),
    ("This situation is making my blood boil",      "Angry"),
    ("I cannot stand this anymore, enough",         "Angry"),
    ("They always take me for granted and it angers me", "Angry"),
    ("I am very annoyed at the constant disrespect","Angry"),
    ("Why does nobody ever listen to what I say",   "Angry"),
    ("I am so angry I cannot even think straight",  "Angry"),
    ("This injustice makes me furious beyond words","Angry"),
    ("I feel enraged every time they do this",      "Angry"),
    ("That was rude and it made me very upset",     "Angry"),

    # ── FEAR ─────────────────────────────────────────────────────
    ("I am terrified of what might happen",         "Fear"),
    ("I feel scared and very anxious about this",   "Fear"),
    ("My hands are shaking with fear and worry",    "Fear"),
    ("I am afraid of what the future holds for me", "Fear"),
    ("Something feels very wrong and I am nervous", "Fear"),
    ("I have a deep sense of dread about tomorrow", "Fear"),
    ("I cannot sleep because of my anxiety",        "Fear"),
    ("I feel panicked and cannot calm down at all", "Fear"),
    ("The thought of failing terrifies me deeply",  "Fear"),
    ("I am overwhelmed with worry and fear",        "Fear"),
    ("I feel uneasy and frightened about this",     "Fear"),
    ("Every little sound at night scares me",       "Fear"),
    ("I am nervous about the results coming out",   "Fear"),
    ("I dread having to face this situation",       "Fear"),
    ("I feel unsafe and very vulnerable right now", "Fear"),

    # ── SURPRISED ────────────────────────────────────────────────
    ("I cannot believe this just happened",              "Surprised"),
    ("This is totally unexpected and shocking",          "Surprised"),
    ("Wow I never saw this coming at all",               "Surprised"),
    ("I am absolutely stunned by this news",             "Surprised"),
    ("This completely blew my mind today",               "Surprised"),
    ("I was not expecting this at all",                  "Surprised"),
    ("What a shocking turn of events this is",           "Surprised"),
    ("I am amazed and astonished by this result",        "Surprised"),
    ("This is unbelievable and I am speechless",         "Surprised"),
    ("Nobody told me this was going to happen",          "Surprised"),
    ("I am shocked beyond words right now",              "Surprised"),
    ("This news completely took me by surprise",         "Surprised"),
    ("I never expected this outcome at all",             "Surprised"),
    ("What a surprise this turned out to be",            "Surprised"),
    ("I am astounded that this actually worked",         "Surprised"),

    # ── NEUTRAL ──────────────────────────────────────────────────
    ("I went to the market today",                       "Neutral"),
    ("The weather is okay today",                        "Neutral"),
    ("I had lunch at twelve o clock",                    "Neutral"),
    ("I read a book in the afternoon",                   "Neutral"),
    ("The meeting is scheduled for tomorrow",            "Neutral"),
    ("I completed my work for the day",                  "Neutral"),
    ("The bus was on time this morning",                 "Neutral"),
    ("I sent the email to the team",                     "Neutral"),
    ("The report will be ready by Friday",               "Neutral"),
    ("I watched a documentary last night",               "Neutral"),
    ("I need to buy some groceries today",               "Neutral"),
    ("The office was quiet today",                       "Neutral"),
    ("I finished reading the chapter",                   "Neutral"),
    ("Today is Wednesday",                               "Neutral"),
    ("I will call you back in some time",                "Neutral"),
    ("The project deadline is next week",                "Neutral"),
    ("I have a class at nine in the morning",            "Neutral"),
    ("The library closes at six in the evening",         "Neutral"),
    ("I submitted the assignment this afternoon",        "Neutral"),
    ("We have a team meeting scheduled on Thursday",     "Neutral"),

    # ── EXTRA HAPPY ──────────────────────────────────────────────
    ("I got selected for the scholarship",               "Happy"),
    ("My family celebrated with a big party",            "Happy"),
    ("I feel incredibly happy and at peace",             "Happy"),
    ("This is the most wonderful surprise ever",         "Happy"),
    ("I am smiling from ear to ear right now",          "Happy"),

    # ── EXTRA SAD ────────────────────────────────────────────────
    ("I feel completely hopeless about my future",       "Sad"),
    ("My best friend moved far away and I miss them",    "Sad"),
    ("I cannot stop the tears from falling",             "Sad"),
    ("Everything I do seems to go wrong lately",         "Sad"),
    ("I feel invisible and completely forgotten",        "Sad"),

    # ── EXTRA ANGRY ──────────────────────────────────────────────
    ("I am seething with anger at what happened",        "Angry"),
    ("They disrespected me and I will not tolerate it",  "Angry"),
    ("I am beyond frustrated with this situation",       "Angry"),
    ("Stop treating me unfairly, I am outraged",         "Angry"),
    ("I am absolutely infuriated by this decision",      "Angry"),

    # ── EXTRA FEAR ───────────────────────────────────────────────
    ("I feel a constant sense of dread and anxiety",     "Fear"),
    ("I am scared of being alone in the dark",           "Fear"),
    ("My heart races every time I think about it",       "Fear"),
    ("I am frightened by all the uncertainty ahead",     "Fear"),
    ("The fear of failure is consuming me completely",   "Fear"),
]


# ================================================================
#  STEP 2: STOPWORDS
#  Stopwords are common words like "the", "is", "am" that carry
#  very little emotional meaning. We remove them so the AI can
#  focus on the words that actually matter for emotion detection.
# ================================================================

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his",
    "she", "her", "hers", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "a", "an", "the", "and", "but", "or", "nor", "for", "so",
    "yet", "both", "either", "neither", "not", "no", "than",
    "too", "very", "just", "because", "as", "until", "while",
    "of", "at", "by", "to", "up", "in", "out", "on", "off",
    "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "more", "most", "other", "some",
    "such", "own", "same", "so", "than", "into", "through",
    "about", "with", "from", "also", "been", "after", "before",
}


# ================================================================
#  STEP 3: TEXT PREPROCESSING FUNCTION
#  Raw text is messy. We clean it up before feeding it to the AI.
#  Steps: lowercase → remove punctuation → remove stopwords
# ================================================================

def preprocess(text: str) -> str:
    """
    Clean and normalize input text.

    Example:
      Input:  "I feel AMAZING today!!!"
      Output: "feel amazing today"
    """
    # 3a. Lowercase — so "Happy" and "happy" are treated the same
    text = text.lower()

    # 3b. Remove punctuation — "amazing!!!" becomes "amazing"
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3c. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 3d. Remove stopwords — split into words, drop stopwords, rejoin
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)


# ================================================================
#  STEP 4: BUILD DATAFRAME
#  We put our dataset into a pandas DataFrame — like a spreadsheet.
#  This makes it very easy to process and pass to scikit-learn.
# ================================================================

def build_dataframe() -> pd.DataFrame:
    """Convert the DATASET list into a pandas DataFrame."""
    df = pd.DataFrame(DATASET, columns=["text", "emotion"])
    # Apply preprocessing to every sentence in the dataset
    df["clean_text"] = df["text"].apply(preprocess)
    return df


# ================================================================
#  STEP 5: TRAIN THE MODEL
#  This is the core ML step. We:
#    a) Convert clean text → TF-IDF numbers
#    b) Train a Logistic Regression classifier
#    c) Save the model to disk so Flask can load it
# ================================================================

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

def train_and_save():
    """
    Train the AI model and save it.
    Called once at startup if no saved model exists.
    """
    print("🤖 Training the AI model...")

    df = build_dataframe()

    # ── 5a. TF-IDF Vectorizer ─────────────────────────────────
    # TF-IDF (Term Frequency–Inverse Document Frequency) converts
    # text into a table of numbers. Each word becomes a column.
    # Words that are rare and meaningful get higher scores.
    # ngram_range=(1,2) means we also consider 2-word pairs like
    # "feel amazing" — which carry more meaning than single words.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # use single words AND word pairs
        max_features=3000,    # only keep the top 3000 features
        sublinear_tf=True,    # apply log scaling to frequencies
    )

    X = vectorizer.fit_transform(df["clean_text"])  # Feature matrix
    y = df["emotion"]                               # Labels (target)

    # ── 5b. Train/Test Split ──────────────────────────────────
    # We hold out 20% of data to test accuracy honestly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 5c. Logistic Regression ───────────────────────────────
    # A powerful yet simple algorithm. It learns the probability
    # that a sentence belongs to each emotion class, then picks
    # the one with the highest probability.
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X_train, y_train)

    # ── 5d. Evaluate ──────────────────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained! Test Accuracy: {accuracy * 100:.1f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── 5e. Save to disk ──────────────────────────────────────
    with open(MODEL_PATH, "wb")      as f: pickle.dump(model,      f)
    with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)
    print("💾 Model saved!\n")

    return model, vectorizer


def load_model():
    """Load the saved model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    with open(MODEL_PATH, "rb")      as f: model      = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    return model, vectorizer


# ================================================================
#  STEP 6: PREDICTION + EXPLANATION
#  The heart of the user experience:
#    - Predict the emotion
#    - Show confidence percentages for all emotions
#    - Explain WHICH words drove the prediction
# ================================================================

# Emoji map — makes output friendly and visual
EMOJI = {
    "Happy":     "😊",
    "Sad":       "😢",
    "Angry":     "😠",
    "Fear":      "😨",
    "Surprised": "😲",
    "Neutral":   "😐",
}

# Color codes for terminal output
COLORS = {
    "Happy":     "\033[92m",   # Green
    "Sad":       "\033[94m",   # Blue
    "Angry":     "\033[91m",   # Red
    "Fear":      "\033[95m",   # Magenta
    "Surprised": "\033[93m",   # Yellow
    "Neutral":   "\033[97m",   # White
    "RESET":     "\033[0m",
    "BOLD":      "\033[1m",
}


def predict_mood(text: str, model=None, vectorizer=None):
    """
    Predict the emotion in a sentence and explain the reasoning.

    Returns a dict with:
      - emotion       : the predicted label ("Happy", "Sad", etc.)
      - emoji         : the matching emoji
      - confidence    : confidence % for the top prediction
      - all_probs     : dict of {emotion: confidence%} for all classes
      - key_words     : list of important words that drove the prediction
      - clean_text    : the preprocessed version of input
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    # Clean the input the same way we cleaned training data
    clean = preprocess(text)

    if not clean.strip():
        return {
            "emotion":    "Neutral",
            "emoji":      EMOJI["Neutral"],
            "confidence": 100.0,
            "all_probs":  {"Neutral": 100.0},
            "key_words":  [],
            "clean_text": clean,
        }

    # Convert text to TF-IDF numbers (same vectorizer as training)
    X = vectorizer.transform([clean])

    # Get predicted class and probabilities for ALL emotions
    emotion   = model.predict(X)[0]
    probs     = model.predict_proba(X)[0]
    classes   = model.classes_

    all_probs = {
        cls: round(prob * 100, 1)
        for cls, prob in sorted(zip(classes, probs),
                                key=lambda x: -x[1])
    }

    # ── Explanation: find key words ───────────────────────────
    # Each word in the vocabulary has a coefficient (weight) for
    # each emotion class. Positive = pushes toward that emotion.
    # We find which words in the input have the highest weight
    # for the predicted emotion — those are the "key words".
    feature_names = vectorizer.get_feature_names_out()
    class_index   = list(classes).index(emotion)
    coefficients  = model.coef_[class_index]

    # Get TF-IDF scores for the words in this sentence
    tfidf_scores = X.toarray()[0]

    # Score = TF-IDF weight × model coefficient (importance)
    word_scores = []
    for idx, score in enumerate(tfidf_scores):
        if score > 0:
            word_scores.append((
                feature_names[idx],
                score * coefficients[idx]
            ))

    # Sort by importance, take top 5 positive contributors
    word_scores.sort(key=lambda x: -x[1])
    key_words = [w for w, s in word_scores if s > 0][:5]

    return {
        "emotion":    emotion,
        "emoji":      EMOJI.get(emotion, "🤔"),
        "confidence": all_probs[emotion],
        "all_probs":  all_probs,
        "key_words":  key_words,
        "clean_text": clean,
    }


# ================================================================
#  STEP 7: COMMAND-LINE INTERFACE
#  Simple interactive terminal session — type a sentence,
#  get back the predicted mood with a detailed explanation.
# ================================================================

def print_result(result: dict, original_text: str):
    """Pretty-print the prediction result in the terminal."""
    C = COLORS
    emotion = result["emotion"]
    color   = C.get(emotion, C["RESET"])
    reset   = C["RESET"]
    bold    = C["BOLD"]

    print("\n" + "═" * 55)
    print(f"  Input : {original_text}")
    print(f"  Cleaned: {result['clean_text']}")
    print("─" * 55)
    print(f"  {bold}Predicted Mood : "
          f"{color}{result['emoji']} {emotion}{reset}")
    print(f"  Confidence     : {color}{result['confidence']}%{reset}")
    print("─" * 55)

    # Confidence bar chart for all emotions
    print(f"  {bold}All Emotion Scores:{reset}")
    for emo, pct in result["all_probs"].items():
        bar_len = int(pct / 5)          # Each █ = 5%
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        emo_col = COLORS.get(emo, reset)
        print(f"    {emo:<10} {emo_col}{bar}{reset} {pct}%")

    # Key words explanation
    if result["key_words"]:
        print("─" * 55)
        print(f"  {bold}Why this mood?{reset}")
        print(f"  Key words → {color}{', '.join(result['key_words'])}{reset}")
        print(f"  These words strongly signal '{emotion}' to the AI.")

    print("═" * 55 + "\n")


def run_cli():
    """Run the interactive command-line mood detector."""
    model, vectorizer = load_model()

    print("\n" + "═" * 55)
    print("       🧠 AI MOOD DETECTOR FROM TEXT")
    print("       Powered by NLP + Logistic Regression")
    print("═" * 55)
    print("  Type any sentence and I'll detect the emotion.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("  📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            break

        if not text:
            print("  ⚠  Please type something.\n")
            continue

        if text.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye! Keep exploring AI.\n")
            break

        result = predict_mood(text, model, vectorizer)
        print_result(result, text)


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    # If no saved model exists, train first
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    run_cli()
