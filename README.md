# 📧 Email Spam Classifier

Un sistema end-to-end per classificare email spam/ham, estrarre topic ricorrenti, misurare la distanza semantica tra i due corpora ed identificare le organizzazioni citate nelle email legittime.

---

## 📋 Indice

- [Descrizione del Progetto](#-descrizione-del-progetto)
- [Architettura](#-architettura)
- [Requisiti](#-requisiti)
- [Installazione](#-installazione)
- [Struttura della Repository](#-struttura-della-repository)
- [Utilizzo](#-utilizzo)
- [Moduli](#-moduli)
- [Dataset](#-dataset)
- [Risultati](#-risultati)
- [Notebook](#-notebook)

---

## 🎯 Descrizione del Progetto

Il progetto implementa una pipeline NLP completa composta da quattro fasi:

| # | Fase | Descrizione |
|---|------|-------------|
| 1 | **Text Classification** | MLP Classifier addestrato su TF-IDF per distinguere email SPAM da HAM (~98% accuracy) |
| 2 | **Topic Modeling** | LDA Multicore (Gensim) per estrarre i 15 topic principali dalle email SPAM e HAM |
| 3 | **Cosine Distance** | Media vettoriale GloVe (300d) per misurare il boundary semantico tra SPAM e HAM |
| 4 | **NER** | Estrazione di organizzazioni (ORG entities) dalle email legittime via spaCy |

---

## 🏗 Architettura

```
email-spam-classifier/
│
├── main.py                    # Entry point CLI
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Pulizia testo, lemmatizzazione, stemming
│   ├── classifier.py          # TF-IDF vectorizer + MLP training/inference
│   ├── topic_modeling.py      # LDA corpus building e training
│   ├── semantic_distance.py   # GloVe + cosine similarity
│   ├── ner_extractor.py       # spaCy NER per organizzazioni
│   └── pipeline.py            # Orchestratore end-to-end
│
├── data/
│   └── spam_dataset.csv       # Dataset originale (label, text, label_num)
│
├── models/                    # joblib artifacts (generati al primo run)
│   ├── mlp_classifier_model.joblib
│   └── vectorizer.joblib
│
├── reports/
│   └── figures/               # PNG charts generati automaticamente dalla pipeline
└── notebooks/
    └── spam_filter.ipynb      # Notebook esplorativo originale del Data Scientist
```

---

## ⚙️ Requisiti

- Python **3.9+**
- pip

> ⚠️ Il download del modello GloVe (`glove-wiki-gigaword-300`) richiede ~1 GB di spazio su disco e avviene automaticamente al primo utilizzo tramite `gensim.downloader`.

---

## 🚀 Installazione

### 1. Clona la repository

```bash
git clone https://github.com/perofficial/email-spam-classifier-topic-modeling-ner.git
cd email-spam-classifier-topic-modeling-ner
```

### 2. Crea un virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# oppure
.venv\Scripts\activate           # Windows
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Scarica il modello spaCy

```bash
python -m spacy download en_core_web_sm
```

---

## 💻 Utilizzo

### Pipeline completa (train + topic modeling + NER)

```bash
python main.py
```

Questo comando:
1. Carica `data/spam_dataset.csv`
2. Preprocessa i testi (lemmatizzazione, stopwords removal, pulizia numeri/punteggiatura)
3. Addestra il classificatore MLP e salva i joblib in `models/`
4. Esegue LDA su SPAM e HAM
5. Scarica GloVe e calcola la cosine distance
6. Estrae le organizzazioni dalle email HAM

### Senza download GloVe (smoke-test rapido)

```bash
python main.py --skip-glove
```

### Classifica una singola email

```bash
python main.py --predict path/to/email.txt
```

Output:
```
Prediction : HAM
Label      : 0
P(HAM)     : 0.9921
P(SPAM)    : 0.0079
```

### Opzioni avanzate

```bash
python main.py \
  --data data/spam_dataset.csv \
  --model models/mlp_classifier_model.joblib \
  --vectorizer models/vectorizer.joblib \
  --num-topics 15
```

### Utilizzo come libreria Python

```python
from src.pipeline import run_pipeline

results = run_pipeline(data_path="data/spam_dataset.csv", skip_glove=True)

# Inferenza su email nuova
from src.classifier import load_artifacts, predict_email
from src.preprocessing import load_nlp_models

model, vectorizer = load_artifacts()
nlp, stop_words = load_nlp_models()

result = predict_email("Win a free iPhone now!!!", model, vectorizer, nlp, stop_words)
print(result["class"])  # → SPAM
```

---

## 📦 Moduli

### `src/preprocessing.py`
Funzioni di pulizia testo condivise da tutti i moduli:
- `clean_text()` — pipeline completa (lowercase → punteggiatura → lemma → stopwords → digit removal)
- `stemming_word()` — Porter stemmer su singola parola
- `sent_to_words()` — tokenizzazione gensim per LDA

### `src/classifier.py`
- `create_bow()` — crea o riusa un TF-IDF vectorizer
- `build_model()` — istanzia MLP con iperparametri di default
- `train()` — split → vectorize → fit → evaluation report
- `predict_email()` — classifica una singola email raw
- `save_artifacts()` / `load_artifacts()` — persistenza joblib

### `src/topic_modeling.py`
- `build_lda_corpus()` — dizionario gensim + BoW corpus
- `train_lda()` — LDA Multicore
- `extract_topic_string()` — appiattisce i topic in un'unica stringa (usata per il cosine distance)

### `src/semantic_distance.py`
- `load_glove()` — download/caricamento GloVe 300d
- `avg_vector()` — vettore medio GloVe di un testo
- `topic_cosine_similarity()` — similarità tra topic aggregati spam vs ham
- `pairwise_mean_cosine_similarity()` — media pairwise su tutti i corpora

### `src/visualizations.py`
Otto grafici generati automaticamente al termine della pipeline e salvati in `reports/figures/`:

| File | Contenuto |
|------|-----------|
| `class_distribution.png` | Bilanciamento classi spam / ham |
| `confusion_matrix.png` | Matrice di confusione |
| `classification_report.png` | Precision / Recall / F1 per classe |
| `roc_curve.png` | Curva ROC con AUC score |
| `training_loss_curve.png` | Loss del MLP durante il training |
| `topics_spam_topics.png` | Top keyword per topic LDA — spam |
| `topics_ham_topics.png` | Top keyword per topic LDA — ham |
| `cosine_similarity.png` | Distanza semantica spam vs ham |

### `src/ner_extractor.py`
- `extract_organisations()` — spaCy NER per entità ORG
- `print_organisations()` — stampa formattata

### `src/pipeline.py`
- `run_pipeline()` — orchestratore end-to-end delle 5 fasi

---

## 📊 Dataset

Il file `data/spam_dataset.csv` contiene:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `label` | str | `"spam"` / `"ham"` |
| `text` | str | Corpo dell'email |
| `label_num` | int | `1` = spam, `0` = ham |

Il dataset è sbilanciato (più HAM che SPAM), situazione tipica nei contesti reali.

---

## 📈 Risultati

| Metrica | Valore |
|---------|--------|
| Accuracy (test set) | ~98% |
| F1-score SPAM | ~98% |
| Topic cosine similarity (spam vs ham aggregato) | ~0.XX |
| **Average pairwise cosine similarity** | **~0.44** |

La similarità media pairwise di ~0.44 indica un buon boundary semantico tra le due classi. Un valore più basso si otterrebbe con ulteriore tuning del preprocessing.

---

## 📓 Notebook

Il notebook esplorativo originale del Data Scientist è disponibile in:

```
notebooks/spam_filter.ipynb
```

Contiene l'intera analisi esplorativa (EDA, esperimenti, visualizzazioni e commenti inline) che ha portato al codice della libreria.

Per avviarlo:

```bash
pip install notebook
jupyter notebook notebooks/spam_filter.ipynb
```

---

## 🤝 Contributing

1. Fork della repository
2. Crea un branch feature (`git checkout -b feature/nome-feature`)
3. Commit delle modifiche (`git commit -m 'feat: descrizione'`)
4. Push del branch (`git push origin feature/nome-feature`)
5. Apri una Pull Request

---

## 📄 License

MIT License — © ProfessionAI
