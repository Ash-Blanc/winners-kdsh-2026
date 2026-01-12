# Winners KDSH 2026 - Track A Solution

## Project Overview

This project implements a solution for the **KDSH 2026 Track A: Constraint Evolution & Causal Compatibility** challenge. The core objective is to determine whether a hypothesized past (backstory) is causally and logically compatible with the observed narrative of a character in a novel.

The system uses a **Hybrid Retrieval (RAG)** approach combined with a **DSPy** pipeline optimized by **Centimators**. It processes novels, builds a document store, and uses an LLM to reason about narrative constraints.

## Key Technologies

*   **Pathway:** For data ingestion, document storage, and hybrid retrieval (Vector + BM25).
*   **DSPy:** For defining the LLM pipeline, signatures, and optimization.
*   **Centimators:** Uses `DSPyMator` for meta-optimization of the DSPy program.
*   **Mistral AI:** Uses `mistral-large-latest` for reasoning and `mistral-embed` for embeddings.
*   **Polars:** For efficient data manipulation.

## Directory Structure

*   `src/submission.py`: The main entry point. Contains the entire pipeline logic (ingestion, indexing, optimization, evaluation).
*   `data/`: Contains the dataset files.
    *   `train.csv`: Training data with backstories and labels.
    *   `test.csv`: Test data for generating predictions.
    *   `gold_result.csv`: Ground truth data for local validation.
*   `novels/`: Directory containing the full text of the novels (e.g., *In Search of the Castaways*, *The Count of Monte Cristo*).
*   `.env`: Configuration file for environment variables.

## Setup & Installation

### Prerequisites

*   Python 3.13+
*   `uv` (recommended) or `pip`

### Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

Alternatively, with pip:

```bash
pip install -e .
```

### Environment Configuration

Ensure you have a `.env` file in the root directory with your Mistral API key:

```env
MISTRAL_API_KEY=your_actual_api_key_here
```

## Usage

To run the full pipeline (optimization, prediction, and evaluation):

```bash
python src/submission.py
```

### Workflow
1.  **Ingestion:** Reads novels from `novels/` using Pathway.
2.  **Indexing:** Splits text and builds a Hybrid Index (Vector + BM25).
3.  **Tuning:** Iterates through different retrieval `k` values (18, 22, 24, 28, 32).
4.  **Optimization:** Uses `DSPyMator` to optimize the `GlobalConsistencyPipeline` on the training set.
5.  **Output:** 
    *   Incremental results are saved as `result_0.csv`, `result_1.csv`, etc.
    *   The best performing configuration (based on accuracy vs `gold_result.csv`) is saved as `best_result.csv`.

## Architecture Details

*   **Ingestion:** `pathway.io.fs.read` streams the novel text.
*   **Chunking:** `TokenCountSplitter` (1000 tokens, 160 overlap).
*   **Retrieval:** Hybrid search weighting Vector (0.7) and BM25 (0.3).
*   **Reasoning:** `ConstraintEvolutionVerifier` signature checks for:
    *   Constraint Evolution
    *   Causal Compatibility
    *   Contradictions
*   **Optimization:** `BootstrapFewShot` via `DSPyMator` tunes the prompt prompts using the training data.
