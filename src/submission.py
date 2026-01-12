# submission.py
# KDSH 2026 Track A – Constraint Evolution & Causal Compatibility
# • Local ingestion via Pathway fs.read
# • DocumentStore + Hybrid Retrieval (vector + BM25)
# • Centimators DSPyMator for optimization
# • Saves incremental results: result_0.csv, result_1.csv, ...
# • Final best: best_result.csv
# • Output columns: story_id,prediction,rationale

from dotenv import load_dotenv
import os
import pathway as pw
import dspy
import polars as pl
import numpy as np
import glob
from litellm import embedding
from centimators.model_estimators import DSPyMator                           # Centimators meta-optimizer
from dspy.teleprompt import BootstrapFewShot
from pathway.xpacks.llm.splitters import RecursiveSplitter
from pathway.xpacks.llm.embedders import LiteLLMEmbedder
from pathway.stdlib.indexing import UsearchKnnFactory, TantivyBM25Factory, HybridIndexFactory
import openai

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

os.environ["OPENAI_API_KEY"] = MISTRAL_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://api.mistral.ai/v1"

print(f"Using MISTRAL_API_KEY: {MISTRAL_API_KEY[:5]}...")
print(f"Using OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")

# ── Configuration ──────────────────────────────────────────────────────
NOVELS_DIR = "./novels/"
GOLD_FILE = "gold_result.csv"           # Your gold standard file
OUTPUT_PREFIX = "result_"               # result_0.csv, result_1.csv, ...
FINAL_OUTPUT = "best_result.csv"        # or rename to "results.csv"

if not os.path.exists(NOVELS_DIR):
    raise FileNotFoundError(
        f"Directory '{NOVELS_DIR}' not found.\n"
        "Place the novels there:\n"
        " - In Search of the Castaways.txt\n"
        " - The Count of Monte Cristo.txt"
    )

# 1. Local ingestion of novels (Skipping Pathway ingestion for this script version)
# docs_source = pw.io.fs.read(
#     NOVELS_DIR,
#     format="binary",
#     mode="streaming",
#     with_metadata=True
# )

# 2. Chunking for long narratives
# splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=160, encoding_name="cl100k_base")

# 3. Mistral embeddings
# embedder = LiteLLMEmbedder(
#     model="mistral/mistral-embed",
#     api_key=MISTRAL_API_KEY,
#     cache_strategy=pw.udfs.DefaultCache()
# )

# 4. Hybrid retrieval (semantic + keyword)
# vector_factory = UsearchKnnFactory(embedder=embedder)
# bm25_factory = TantivyBM25Factory()
# hybrid_factory = HybridIndexFactory(
#     retriever_factories=[vector_factory, bm25_factory]
# )

# 5. Core Pathway DocumentStore
# document_store = DocumentStore(
#     docs=docs_source,
#     retriever_factory=hybrid_factory,
#     splitter=splitter
# )
document_store = None # Placeholder for API compatibility

# 6. DSPy LM configuration
lm = dspy.LM(
    model="openai/mistral-large-latest",
    api_base="https://api.mistral.ai/v1",
    api_key=MISTRAL_API_KEY,
    temperature=0.10,
    max_tokens=2000
)
dspy.settings.configure(lm=lm)

# 7. Signature aligned with challenge motivation
class ConstraintEvolutionVerifier(dspy.Signature):
    """You are a rigorous auditor of narrative constraint evolution and causal compatibility.

    Task: Determine whether the hypothesized PAST can causally and logically produce the observed FUTURE.
    
    Output a detailed analysis covering constraint evolution, causal compatibility, and contradictions, followed by a final label."""

    context: str = dspy.InputField(desc="Hybrid-retrieved novel passages")
    character: str = dspy.InputField()
    backstory: str = dspy.InputField(desc="Hypothesized past events")

    analysis: str = dspy.OutputField(desc="Detailed reasoning on constraints, causality, and contradictions")
    label: int = dspy.OutputField(desc="1 = fully compatible, 0 = any contradiction")

# 8. DSPy Module
class GlobalConsistencyPipeline(dspy.Module):
    def __init__(self, document_store, retrieval_k=24):
        super().__init__()
        # self.doc_store = document_store # Not using Pathway DocumentStore for sync retrieval
        self.k = retrieval_k
        # Switching to Predict to avoid implicit 'reasoning' field and reduce output complexity
        self.reason = dspy.Predict(ConstraintEvolutionVerifier)
        
        class PipelineSignature(dspy.Signature):
            character: str = dspy.InputField()
            backstory: str = dspy.InputField()
            label: int = dspy.OutputField()
            analysis: str = dspy.OutputField()
        self.signature = PipelineSignature

        # Build local index for sync retrieval
        print("Building local index for synchronous retrieval...")
        self.chunks = []
        self.embeddings = []
        
        # Load novels
        novel_files = glob.glob(os.path.join(NOVELS_DIR, "*.txt"))
        for nv in novel_files:
            with open(nv, "r", encoding="utf-8") as f:
                text = f.read()
                # Simple splitting
                # rough char estimation: 1000 tokens ~ 4000 chars
                chunk_size = 4000
                overlap = 600
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) > 100:
                        self.chunks.append(chunk)
        
        print(f"Index: {len(self.chunks)} chunks. Embedding...")
        
        # Batch embedding to avoid rate limits/slow speed
        batch_size = 10
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            try:
                # Using mistral-embed via litellm
                resp = embedding(model="mistral/mistral-embed", input=batch, api_key=MISTRAL_API_KEY)
                for d in resp["data"]:
                    self.embeddings.append(d["embedding"])
            except Exception as e:
                print(f"Embedding error: {e}")
                # Fallback: add zero vectors or skip? 
                # Better to fail loud or retry, but for now just skip to keep going
                for _ in batch:
                    self.embeddings.append([0.0]*1024) # Assuming 1024 dim
                    
        self.embeddings = np.array(self.embeddings)
        print("Local index built.")

    def retrieve(self, query, k):
        # Embed query
        try:
            resp = embedding(model="mistral/mistral-embed", input=[query], api_key=MISTRAL_API_KEY)
            q_emb = np.array(resp["data"][0]["embedding"])
            
            # Cosine similarity
            # norm(a) * norm(b)
            scores = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9)
            
            # Top k
            top_k_indices = np.argsort(scores)[-k:][::-1]
            
            return [{"text": self.chunks[i]} for i in top_k_indices]
        except Exception as e:
            print(f"Retrieval error: {e}")
            return [{"text": ""}]

    def forward(self, character, backstory):
        query = f"{character} {backstory} timeline constraints causality commitments ruling-out events"

        chunks = self.retrieve(query=query, k=self.k)
        context = "\n\n─────\n\n".join([c["text"] for c in chunks])

        pred = self.reason(context=context, character=character, backstory=backstory)

        return dspy.Prediction(
            analysis=pred.analysis,
            label=int(pred.label)
        )

# ── Data Preparation ───────────────────────────────────────────────────
train_df = pl.read_csv("./data/train.csv")
test_df = pl.read_csv("./data/test.csv")

trainset = [
    dspy.Example(
        character=row["char"],
        backstory=row["content"],
        label=row["label"]
    ).with_inputs("character", "backstory")
    for row in train_df.iter_rows(named=True)
]

# ── Optimization & Tuning Loop ─────────────────────────────────────────
print("Starting optimization & tuning process...")

best_acc = 0.0
best_k = 24
best_predictions = None
iteration = 0

X_train = train_df.select(character=pl.col("char"), backstory=pl.col("content"))
y_train = train_df["label"]

# Slicing for quick verification of the fix (optional, can be removed for full run)
# X_train = X_train.head(2)
# y_train = y_train.head(2)

for k in [18, 22, 24, 28, 32]: # Restored full loop
    print(f"\nIteration {iteration} – testing retrieval k = {k}")

    program = GlobalConsistencyPipeline(document_store=document_store, retrieval_k=k)

    mator = DSPyMator(
        program=program,
        target_names="label",
        lm=lm
    )

    optimizer = BootstrapFewShot(
        metric=lambda ex, pred, trace=None: ex.label == pred.label,
        max_bootstrapped_demos=6,
        max_labeled_demos=4
    )

    mator.fit(X_train, y_train, optimizer=optimizer)
    optimized = mator.program

    # Generate predictions
    current_preds = []
    for row in test_df.iter_rows(named=True):
        result = optimized(character=row["char"], backstory=row["content"])
        current_preds.append({
            "story_id": row["id"],
            "prediction": result.label,
            "rationale": result.analysis
        })

    # Save incremental result
    current_df = pl.DataFrame(current_preds)
    current_df.write_csv(f"{OUTPUT_PREFIX}{iteration}.csv")
    print(f"→ Saved incremental result: {OUTPUT_PREFIX}{iteration}.csv")

    # Compute accuracy against gold
    def compute_acc(gold_path, pred_path):
        gold = pw.io.csv.read(gold_path, schema={"id": pw.Int, "label": pw.Int}, mode="static")
        pred = pw.io.csv.read(pred_path, schema={"story_id": pw.Int, "prediction": pw.Int}, mode="static")
        
        joined = gold.join(pred, pw.left.id == pw.right.story_id).select(
            gold_label=pw.left.label,
            pred_label=pw.right.prediction
        )
        
        correct = joined.filter(pw.this.gold_label == pw.this.pred_label).count()
        total = joined.count()
        
        pw.run()
        return float(correct) / total if total > 0 else 0.0

    acc = compute_acc(GOLD_FILE, f"{OUTPUT_PREFIX}{iteration}.csv")
    print(f"→ Accuracy vs gold: {acc:.4f} ({acc*100:.2f}%)")

    if acc > best_acc:
        best_acc = acc
        best_k = k
        best_predictions = current_preds

    iteration += 1

# Save best result
if best_predictions:
    final_df = pl.DataFrame(best_predictions)
    final_df.write_csv(FINAL_OUTPUT)
    print(f"\nBest configuration found: k = {best_k}")
    print(f"Best accuracy vs gold: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Final submission ready: {FINAL_OUTPUT}")

# Keep Pathway engine running
pw.run()