# CAVA: Causal Analysis and Validation Agent for Microscopy-Informed Materials Discovery

This repository implements a modular, agentic framework for **literature-grounded causal reasoning**, designed to bridge the gap between **data-driven causal discovery** and **established scientific knowledge**. By integrating Large Language Models (LLMs) with automated bibliographic retrieval, the system evaluates and validates directed causal relationships \(X --> Y\) through explicit evidence extracted from peer-reviewed scientific literature.

---

## Methodology Overview

The system follows a multi-stage, fully automated pipeline that transforms causal hypotheses into evidence-backed causal graphs:

1. **Context-Aware Query Generation**  
   A GPT-4–class LLM (**gpt-4o-mini**) acts as a query-generation agent, converting an ordered variable pair \((X, Y)\) into a concise, causality-focused arXiv search query. Prompts emphasize intervention-oriented keywords such as *causal effect*, *mechanism*, *intervention*, and *impact* to prioritize experimental and mechanistic studies.

2. **Automated Literature Retrieval (arXiv)**  
   The generated query is executed against the **arXiv API**, retrieving a ranked set of relevant scientific papers. For each paper, the system collects structured metadata including title, abstract, arXiv ID, DOI (when available), and publication date.

3. **LLM-Driven Causal Evidence Extraction**  
   Abstracts are sentence-tokenized and analyzed by a second LLM agent using a **strict intervention-based causal heuristic**. A sentence is extracted *only if* it explicitly states that manipulating or changing \(X\) causes, increases, decreases, or determines \(Y\). Sentences describing correlations, associations, or inferred downstream effects are explicitly rejected. All outputs are returned in **strict JSON format** to ensure reproducibility and reliable downstream parsing.

4. **Evidence-Backed Causal Graph Construction**  
   Accepted causal sentences are treated as atomic evidence and stored with full bibliographic traceability. These are used to construct a **directed causal graph (DAG)** using **NetworkX**, where each edge \(X \rightarrow Y\) is directly supported by one or more cited literature statements.

5. **Reasoning and Uncertainty Assessment**  
   The system synthesizes a final verdict (e.g., *YES (literature-supported)* or *NO DIRECT ABSTRACT EVIDENCE*) based on the presence of supporting edges. Each verdict includes cited sentences, a summary of searched papers, and an explicit uncertainty statement acknowledging limitations such as abstract-only analysis and lack of full-text or study-design validation.

---

## Technical Features

- **Strict Causal Heuristic**  
  Enforces mandatory causal verbs (e.g., *causes, induces, controls, drives*) and requires explicit mention of both cause and effect variables.

- **Agentic LLM Design**  
  Distinct LLM agents are used for query generation and evidence extraction, each governed by tightly scoped prompts to reduce hallucination and improve interpretability.

- **Multi-Algorithm Comparison**  
  Literature-derived causal graphs can be directly compared with graphs learned from automated causal discovery algorithms such as **PC, FCI, and GRaSP**, loaded from CSV outputs.

- **Visualization Suite**  
  Includes custom plotting utilities to render clean, interpretable DAGs with consistent layouts, arrowheads, and labeled nodes for both literature-derived and algorithmic graphs.

- **Extensible Architecture**  
  The pipeline is domain-agnostic and can be reused across scientific fields by modifying only the input variable pairs.

---

## Setup and Usage

### Requirements
- Python ≥ 3.9  
- Core dependencies:
  - `openai`
  - `arxiv`
  - `networkx`
  - `matplotlib`
  - `pandas`
  - `tqdm`

### LLM Configuration
An OpenAI API key is required for query generation and evidence extraction:
```python
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

