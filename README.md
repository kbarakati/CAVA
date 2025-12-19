# CAVA: Causal Analysis and Validation Agent for Microscopy-Informed Materials Discovery

## 🎥 Video Presentation
For a concise overview of the motivation, methodology, and results of **CAVA**, please watch the accompanying video presentation:

👉 **YouTube:** https://youtu.be/qxorrjfsiOw

The video introduces the agentic workflow, literature-grounded causal reasoning pipeline, and representative outputs, and is recommended for first-time readers before exploring the codebase.

---

## Overview

**CAVA** is a modular, agentic framework for **literature-grounded causal reasoning**, designed to bridge data-driven causal discovery with established scientific knowledge. The system integrates **automated causal discovery**, **large language model (LLM) agents**, and **bibliographic retrieval** to evaluate whether directed causal relationships \(X \rightarrow Y\) are explicitly supported by prior scientific literature.

CAVA is motivated by microscopy-informed materials discovery, where understanding relationships among composition, structure, and properties is essential for experimental interpretation and decision-making. Rather than replacing physics-based models or domain expertise, CAVA provides a **transparent, evidence-backed reference** that contextualizes data-derived causal relationships within the existing literature.

---

## Methodology Overview

CAVA follows a fully automated, multi-stage pipeline that transforms ordered variable pairs into evidence-supported causal graphs.

### 1. Context-Aware Query Generation
A GPT-4–class LLM (**gpt-4o-mini**) acts as a query-generation agent. Given an ordered variable pair \((X, Y)\), the agent constructs a concise, causality-focused arXiv search query. Prompts emphasize intervention-oriented keywords such as *causal effect*, *mechanism*, *intervention*, and *impact* to prioritize experimental and mechanistic studies.

### 2. Automated Literature Retrieval (arXiv)
The generated query is executed against the **arXiv API**, retrieving a ranked set of relevant scientific papers. For each paper, the system collects structured metadata including title, abstract, arXiv ID, DOI (when available), and publication date.

### 3. LLM-Driven Causal Evidence Extraction
Abstracts are sentence-tokenized and analyzed by a second LLM agent using a **strict intervention-based causal heuristic**. A sentence is extracted *only if* it explicitly states that manipulating or changing \(X\) causes, increases, decreases, or determines \(Y\). Sentences describing correlations, associations, or inferred downstream effects are explicitly rejected. All outputs are returned in **strict JSON format** to ensure reproducibility and reliable downstream parsing.

### 4. Evidence-Backed Causal Graph Construction
Accepted causal sentences are treated as atomic evidence and stored with full bibliographic traceability. These are used to construct a **directed causal graph (DAG)** using **NetworkX**, where each edge \(X \rightarrow Y\) is directly supported by one or more cited literature statements.

### 5. Reasoning and Uncertainty Assessment
The system synthesizes a final verdict (e.g., *YES (literature-supported)* or *NO DIRECT ABSTRACT EVIDENCE*) based on the presence of supporting edges. Each verdict includes cited sentences, a summary of searched papers, and an explicit uncertainty statement acknowledging limitations such as abstract-only analysis and the absence of full-text or study-design validation.

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

### Installation
```bash
pip install -r requirements.txt
