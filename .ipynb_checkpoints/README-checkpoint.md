# Price Transparency Llama-RAG

Explainable hospital cost estimation using Retrieval-Augmented Generation.

This system estimates medical procedure cost using real hospital negotiated prices
and explains how the estimate was calculated.

---

## Process
You ask a question:

"I need a colonoscopy near Pittsburgh with UPMC insurance"

The system:
1) Understands your medical intent (Llama)
2) Finds relevant CPT billing codes
3) Retrieves real hospital prices
4) Calculates weighted cost
5) Explains the estimate

---

## Cost Calculation

Estimated Cost = Σ (Wi × Mi)

Wi = relevance weight from Llama  
Mi = median negotiated hospital price

---

## Confidence Levels

Tier 1 — exact payer negotiated rate
Tier 2 — hospital median rate
Tier 3 — global median fallback

---

## How to Run

python src/price_llama_rag.py

---

## Important
Dataset is not uploaded due to size and privacy.
Model works only when local CSV is provided.