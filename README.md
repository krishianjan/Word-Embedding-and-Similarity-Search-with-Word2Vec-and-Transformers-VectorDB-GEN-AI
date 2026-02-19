📘 Project Overview  This project demonstrates the mathematical and visual foundation of modern GenAI and LLM applications — focusing on word embeddings, similarity computations, and contextual relationships using LLama Model,  Word2Vec, CBOW, and Skip-Gram models.
It implements a semantic search pipeline capable of learning linguistic patterns, ranking contextual closeness, and visualizing high-dimensional vector spaces through PCA/t-SNE projections and heatmaps.

The system serves as a miniature Retrieval-Augmented Generation (RAG) foundation, where vectorized text representations can be later integrated into GenAI workflows for document search, intent classification, or email categorization.

🚀 Features

✅ Custom Preprocessing Pipeline

Text cleaning, tokenization, lemmatization, and stopword removal using NLTK

Contextual normalization to improve semantic vector consistency

✅ Dynamic Word Embedding Training

Trained multiple Word2Vec models (CBOW/Skip-gram) with variable hyperparameters:

Vector dimensions (10, 50, 100)

Context window sizes (3, 5, 10)

Training epochs (1, 10, 100)

Comparative analysis of embedding stability and convergence

✅ Semantic Similarity Analytics

Quantifies cosine similarity between key concept pairs (e.g., machine-learning, neural-network, artificial-intelligence)

Evaluates embedding quality under different hyperparameters

Generates bar and line charts for vector space evaluation metrics

✅ 2D Vector Visualization (PCA & t-SNE)

Transforms embeddings into 2D plots to visualize clustering of semantically related words

Identifies relationship strength between tokens using heatmaps

✅ Cross-Corpus Embedding Extension

Integrates new domains (React, JS, Redux corpus) into existing model

Performs transfer-learning style embedding update and retraining

Evaluates cross-domain word similarity and semantic drift

🧮 Math & Logic Behind the Model

Word2Vec maps each word w into a vector v(w) in Rⁿ

The CBOW model predicts a target word from its context window

The Skip-Gram model predicts context words from a target word

Cosine similarity: S𝑖𝑚(𝑎,𝑏)=𝑎⋅𝑏∥𝑎∥∥𝑏∥  sim(a,b)=  ∥a∥∥b∥ a⋅b
	​


measures how close two words are semantically

PCA and t-SNE reduce high-dimensional embeddings to 2D while preserving cluster proximity

📊 Visualizations

Embedding Scatter Plots (PCA / t-SNE) – showing semantic grouping of similar words

Heatmaps – display pairwise similarity intensities

Line Charts – track similarity variations across dimensions, epochs, and context windows

🧩 Tech Stack

Language: Python

Libraries: Gensim, NLTK, NumPy, scikit-learn, Matplotlib, re

Tech Concepts: Word Embeddings, NLP, Vector Similarity, PCA/t-SNE, RAG

Deployment: Modularized to integrate into GenAI pipelines (e.g., Google Agentspace, LangChain, or Vertex AI)

🧠 Key Learnings

Built a complete embedding-based semantic reasoning pipeline from scratch

Tuned model hyperparameters to optimize embedding quality

Visualized vector spaces to interpret model understanding

Bridged mathematical NLP theory with applied GenAI foundation design
