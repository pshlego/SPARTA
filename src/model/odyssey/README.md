# Odyssey Pipeline

Odyssey is a multi-hop QA model that reasons over table-text hybrid data using graph traversal.
It runs as an 8-step sequential pipeline, where each step produces intermediate outputs consumed by later steps.

## Quick Start

Run the full pipeline at once:

```bash
# Single process
sh scripts/odyssey/run_all.sh benchmark=sparta benchmark.domain=nba model=odyssey model.use_accelerate=false

# Multi-process (8 GPUs)
sh scripts/odyssey/run_all.sh benchmark=sparta benchmark.domain=nba model=odyssey
```

Available SPARTA domains: `nba`, `movie`, `medical`

## Pipeline Steps

Each step can also be run individually. Steps must be executed **in order** — each step depends on the outputs of previous steps.

### Step 1: Entity Extraction

Extracts named entities from questions using GPT.

```bash
python src/model/odyssey/modules/entity_extraction.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `data/sparta/odyssey/preprocess/ee_predictions.json`

### Step 2: Header Extraction

Selects relevant table headers for each question. Depends on Step 1.

```bash
python src/model/odyssey/modules/header_extraction.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `data/sparta/odyssey/preprocess/he_predictions.json`

### Step 3: Entity-Document Graph Construction

Builds entity-document graphs from passages using spaCy NER. Independent of Steps 1-2 (can run in parallel).

```bash
python src/model/odyssey/modules/make_ent_doc_graphs.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `data/sparta/odyssey/preprocess/entity_doc_graph/`

### Step 4: Table Encoding

Encodes table cells and headers using INSTRUCTOR embeddings. Independent of Steps 1-2 (can run in parallel).

```bash
python src/model/odyssey/modules/table_encoding.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `{benchmark.cache_dir}/table_embeddings/`

### Step 5: Entity Encoding

Encodes entity nodes from entity-document graphs using INSTRUCTOR embeddings. Depends on Step 3.

```bash
python src/model/odyssey/modules/entity_encoding.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `{benchmark.cache_dir}/entity_embeddings/`, `{benchmark.cache_dir}/entity_nodes/`

### Step 6: Entity-to-Header Mapping

Maps extracted entities to table headers using GPT. Depends on Steps 1-2.

```bash
python src/model/odyssey/modules/entity_to_header_mapping.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `data/sparta/odyssey/preprocess/e2h_predictions.json`

### Step 7: Start Point Initialization

Finds graph traversal start points via cosine similarity. Depends on Steps 1-6.

```bash
python src/model/odyssey/modules/init_start_points.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `data/sparta/odyssey/preprocess/start_points.json`

### Step 8: Inference

Multi-hop QA reasoning (up to 3 hops) using GPT. Depends on Steps 1-7.

```bash
python src/model/odyssey/modules/inference.py benchmark=sparta benchmark.domain=nba model=odyssey
```

Output: `data/sparta/odyssey/results/base/qa_results.json`

## Dependency Graph

```
Step 1 (EE) ──┬──→ Step 2 (HE) ──┬──→ Step 6 (E2H) ──┐
              │                   │                     │
              │                   └─────────────────────┤
              │                                         ▼
Step 3 (EDG) ─┬──→ Step 5 (EntEnc) ──────────→ Step 7 (StartPts) ──→ Step 8 (Inference)
              │                                         ▲
Step 4 (TblEnc) ────────────────────────────────────────┘
```

Steps that can run in parallel:
- Steps 1, 3, 4 are independent of each other
- Step 2 needs only Step 1
- Step 5 needs only Step 3
- Step 6 needs Steps 1, 2

## Configuration

All configuration is managed via Hydra. Key options:

| Option | Description |
|--------|-------------|
| `benchmark=sparta` | SPARTA benchmark |
| `benchmark.domain=nba\|movie\|medical` | SPARTA domain |
| `model=odyssey` | Use Odyssey model config |
| `model.use_accelerate=false` | Disable multi-processing (single process) |
| `model.modelName=gpt-3.5-turbo` | LLM model name |
| `model.simThreshold=0.8` | Cosine similarity threshold for start points |
| `model.temperature=0.0` | LLM temperature |

## Multi-GPU Execution

By default (`use_accelerate=true`), scripts use HuggingFace Accelerate for distributed processing.
Use `accelerate launch` to control the number of processes:

```bash
accelerate launch --num_processes=4 src/model/odyssey/modules/entity_extraction.py benchmark=sparta benchmark.domain=nba model=odyssey
```
