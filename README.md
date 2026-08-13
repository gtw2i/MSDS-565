# MSDS 565 — Applied Machine Learning (Fall 2026)

Course materials for **MSDS 565** at Meharry Medical College: the demo notebooks we work through in class, plus the prompt for each of the three group projects. This repository is read-only for you — it is published from the instructor's working copy, so please don't open pull requests against it. Your team's project work lives in a repository your team creates.

The course runs Tuesdays 5:30–8:30 Central. Three group projects, one per unit, each a five-notebook pipeline built in your team's own repository and reported on live in class.

---

## Setting up your environment

Every notebook here runs in a conda environment named `msds565`.

1. Read **`Creating Custom Python Environments.docx`** — it walks through installing and managing conda environments from scratch.
2. Build the environment from the pinned spec in this repository:

```bash
conda env create -f environment.yml
conda activate msds565
```

3. Register it with Jupyter so the notebooks find the right kernel:

```bash
python -m ipykernel install --user --name msds565 --display-name msds565
```

If a notebook fails on an import, check the kernel selected in the top-right of Jupyter before anything else — it should read `msds565`.

### `msds565_helpers.py`

Shared helper functions used across the demos (for example `helpers.train_and_evaluate(...)`) live in **`msds565_helpers.py`** at the root of this repository. The demo notebooks sit two folders below the root and reach it with:

```python
import sys
sys.path.append('../..')
import msds565_helpers as helpers
```

Keep that relative path working — if you move a notebook somewhere else, the import breaks. If you copy a demo notebook into your own project repository, copy `msds565_helpers.py` along with it and fix the path.

---

## Layout

```
MSDS 565/
├── environment.yml                        # the msds565 conda environment
├── msds565_helpers.py                     # shared helpers, imported by the demos
├── Creating Custom Python Environments.docx
├── Unit 1 - Tabular Data/
│   ├── Demos/                             # 41 notebooks — regression and classification
│   └── Project/P1_Prompt.md
├── Unit 2 - Image Data/
│   ├── Demos/                             # 10 notebooks — image processing and CNNs
│   ├── Extra/                             # a Wikipedia image-scraping utility
│   └── Project/                           # P2_Prompt.md + the DDGS image scraper
└── Unit 3 - Text Data/
    ├── Demos/                             # 14 notebooks — NLP, LLMs, scraping, Streamlit
    ├── Extra/                             # 17 optional notebooks continuing each section
    └── Project/P3_Prompt.md
```

`Extra/` folders are side material — genuinely useful, but not part of the sequence we cover in class and not assumed by any project. Their filenames continue each section's numbering, so `U3-1_NLP-6_WordEmbeddings.ipynb` picks up where the taught `U3-1_NLP-5_arXiv.ipynb` leaves off. Several of them need a local Ollama install or a large model download, which is part of why they sit outside the main sequence.

---

## The three units

### Unit 1 — Tabular Data

Two end-to-end CRISP-DM arcs, each following one dataset the whole way through: a **regression** arc on real-estate pricing (`U1_RealEstate-1…6`) and a **classification** arc on hospital readmission (`U1_Diabetes-1…7`). The numbered `U1-*` notebooks alongside them each isolate one concept.

| Prefix | Topic |
|---|---|
| `U1-1_*` | Intro and EDA — embeddings and vectorization |
| `U1-2_*` | Preprocessing |
| `U1-3_*` | Regression — model comparison, overfitting, cross-validation, grid search, Keras |
| `U1-4_*` | Feature selection — VIF, feature importance, PCA, dimensionality reduction, Lasso |
| `U1-6_*` | Classification — decision regions, entropy, F1, parameter sweeps, Keras |
| `U1-7_*` | Imbalanced classification — stratified K-fold, resampling, SMOTE, class weights |

**Libraries:** scikit-learn, pandas, numpy, Keras/TensorFlow, imbalanced-learn, SHAP, Fairlearn

The datasets for these demos are not committed here — each notebook names its source or its loader, and the two preprocessed CSVs are produced by running the earlier notebooks in the arc.

### Unit 2 — Image Data

The ten demos are **one continuous arc**: each notebook's review section names the next one, so read them in order. `U2-1_Images-*` covers classical, non-learned image processing with scikit-image and OpenCV — the hand-designed filters a CNN later learns for itself. `U2-2_CNN-1…9` then runs from why dense networks fail on images, through MNIST, class imbalance, CIFAR-10, multimodal input and transfer learning, to embeddings, explainability and a fairness audit.

> **The cat/dog image set is not in this repository.** Several Unit 2 demos load images from `Unit 2 - Image Data/Demos/images/cat/` and `.../dog/`. Those files are distributed separately on Blackboard for copyright reasons — download them and unzip into `Unit 2 - Image Data/Demos/` so the `images/cat/` and `images/dog/` folders sit next to the notebooks. Until you do, the image-loading cells will raise a file-not-found error. Two demos (`U2-2_CNN-8_Explainability` and `U2-2_CNN-9_FairnessAudit`) use larger external datasets instead and say so in their opening cell.

These notebooks are stored with their outputs cleared, so you will see the results by running them rather than by reading them on GitHub.

**Libraries:** Keras/TensorFlow, scikit-image, OpenCV, fastembed (CLIP), Fairlearn

### Unit 3 — Text Data

Four sections: NLP fundamentals, LLMs, web scraping, and Streamlit web apps.

| Prefix | Topic |
|---|---|
| `U3-1_NLP-*` | Tokenization, vectorization, text classification with SHAP, corpus work, arXiv |
| `U3-2_LLM-*` | Provider SDKs side by side — basics, JSON schema output, LangChain RAG |
| `U3-3_Scrape-*` | BeautifulSoup, the Wikipedia API, scraping one article and a table of linked pages, news sentiment |
| `U3-4_Streamlit_*.py` | Two runnable Streamlit apps — the reference patterns for the P2 and P3 apps |

The LLM notebooks call **OpenAI, Anthropic, and Google Gemini** side by side and read each key from an environment variable, so you need your own key for whichever provider you run; the local-model notebooks use **Ollama** and cost nothing. `paper_schema.json` is the JSON Schema used for structured LLM output. Run a Streamlit app with `streamlit run "Unit 3 - Text Data/Demos/U3-4_Streamlit_Plotter.py"`.

**Libraries:** NLTK, TextBlob, transformers, BERTopic, LangChain, anthropic, BeautifulSoup, streamlit

---

## Projects

One project per unit, numbered to match it. Each prompt is the authority on its own requirements, repository structure, and filenames — read it in full before you start.

| Project | Unit | Prompt |
|---|---|---|
| P1 | 1 — Tabular | `Unit 1 - Tabular Data/Project/P1_Prompt.md` |
| P2 | 2 — Image | `Unit 2 - Image Data/Project/P2_Prompt.md` |
| P3 | 3 — Text | `Unit 3 - Text Data/Project/P3_Prompt.md` |

Each is a five-notebook group project in a repository your team creates, one notebook per class session, graded from a live in-class progress report rather than from after-the-fact inspection. Any member may be asked to account for any notebook, so everyone works on everything.

P2 ships the tools you build your dataset with, in `Unit 2 - Image Data/Project/`: **`ddgs_scraper.py`** (the scraper module — you import it, you don't rewrite it) and **`ddgs_image_scraper.ipynb`** (the notebook you start P2's first notebook from). Keep the two together in the same folder, since the notebook does `import ddgs_scraper`.

---

## What is not in this repository

On **Blackboard**:

- The project rubric, and the syllabus
- Lecture slide decks
- The peer evaluation form and the meeting minutes template
- The Unit 2 cat/dog image set

On the **class Google Drive**, in the `Datasets` folder:

- The P1 car-sales dataset and the P3 IMDB files — pull data from there rather than downloading it from the original source, so everyone starts from the same copy

If something a notebook or prompt refers to isn't in this repository, check Blackboard and the class Google Drive before asking — and if it isn't in either, ask.
