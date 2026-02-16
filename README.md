# Llama-RAG Hospital Price Transparency

A retrieval-grounded medical cost estimation system using **Llama-2 + semantic search + CMS hospital price transparency data**.

This model requires data which is provided upon request and HuggingFace "meta-llama/Llama-2-7b-chat-hf" token.

Please make sure to insert the token in .py first line 

HUGGINGFACE_TOKEN = "TOKEN-HERE" 

---

## Model Process
Given a natural language question:

> "How much does a colonoscopy cost near Pittsburgh? I have UPMC Health Plan Gold."

The system:
1. Extracts structured context using Llama-2
2. Finds relevant CPT procedure codes
3. Retrieves real hospital negotiated rates
4. Computes a weighted cost estimate
5. Explains how the estimate was calculated

---

## Model Pipeline

### 1. Patient Query
User provides free-text medical cost question.

### 2. Llama Slot Extraction
Llama converts text → structured JSON:

- payer
- plan
- hospitals
- clinical intent

### 3. Semantic Retrieval
SentenceTransformer (`all-MiniLM-L6-v2`) finds closest CPT codes.

### 4. Llama CPT Weighting
Llama acts as a medical coder and assigns weights to the 5 most relevant CPT codes.

Weights sum ≈ 1.0

### 5. Tiered Rate Lookup
For each hospital and CPT, the system retrieves real prices:

| Tier | Source |
|----|----|
| Tier 0.5 | RAG context search |
| Tier 1 | Exact payer + plan negotiated rate |
| Tier 2 | Hospital median rate |
| Tier 3 | Global median rate |

### 6. Cost Calculation

Estimated Cost = Σ (weightᵢ × median_rateᵢ)

### 7. Explanation Output
Model prints:
- CPT codes used
- weights
- insurance context
- hospital cost estimates
- confidence tier

---

## Run

python src/price_llama_rag.py       ## needs csv file in the local system provided with request