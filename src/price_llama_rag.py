HUGGINGFACE_TOKEN = "TOKEN-HERE" 

CSV_PATH = "allUPMC_hospitals_cpt.csv"
RATE_COLS = [
    "standard_charge|negotiated_dollar", "standard_charge|discounted_cash",
    "standard_charge|min", "standard_charge|max", "estimated_amount",
    "standard_charge|gross",
]

# Model used is Llama
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Attempting to use device: {'GPU (Index 0)' if DEVICE == 0 else 'CPU (Index -1)'}")

if HUGGINGFACE_TOKEN == "YOUR_HUGGINGFACE_TOKEN":
    print("FATAL: HUGGINGFACE_TOKEN is not set. Please update the script with your token.")
    sys.exit(1)

_llama_pipe = None
_chroma_collection = None

def _build_llm():
    global _llama_pipe
    if _llama_pipe is None:
        try:
            if torch.cuda.is_available():
                print("[OOM FIX] Clearing CUDA cache...")
                torch.cuda.empty_cache() 
                
            print(f"[LLM] Loading Llama-2-7b-chat-hf in standard torch.float16...")
            llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGINGFACE_TOKEN)
            
            llama_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                token=HUGGINGFACE_TOKEN,
                device_map="auto" if torch.cuda.is_available() else None 
            )

            _llama_pipe = pipeline(
                "text-generation",
                model=llama_model,
                tokenizer=llama_tokenizer
            )
            print("[LLM] Meta-Llama pipeline ready.")
        except Exception as e:
            print(f"[LLM ERROR] FATAL: Could not initialize LLM. Check token/GPU. Error: {e}")
            print(f"HINT: Without quantization, this model requires a GPU with at least 12GB VRAM.")
            raise e

    def gen(prompt: str, max_new_tokens=220):
        prompt_template = f"<s>[INST] {prompt.strip()} [/INST]"
        out = _llama_pipe(
            prompt_template,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_llama_pipe.tokenizer.eos_token_id, 
        )
        generated_text = out[0]["generated_text"].replace(prompt_template, "").strip()
        return generated_text.split("[/INST]")[-1].strip().replace("</s>", "").strip()
    return gen

print("[RAG] Initializing Sentence Transformer for semantic search...")
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2') 
print("[RAG] Sentence Transformer ready.")

def get_rate(rec: pd.Series):
    """Finds the first valid, positive rate value from the prioritized columns."""
    for c in RATE_COLS:
        v = rec.get(c)
        try:
            x = float(v)
            if x > 0: return x
        except: 
            pass
    return np.nan 

def tokenize(t: str): return re.findall(r"[a-z0-9]+", (t or "").lower())

def ask(q): print(f"\nAssistant: {q}"); return input("You: ").strip()

def _best_match(text: str, options: List[str]) -> str:
    """Snaps a free-text name to the closest one in the vocabulary list."""
    if not text.strip(): return ""
        
    s_tok = set(tokenize(text))
    if not s_tok: return ""
    
    best, best_score = "", -1
    
    for opt in options:
        opt_tokens = set(tokenize(opt))
        current_score = len(s_tok & opt_tokens)
        
        if current_score > best_score:
            best, best_score = opt, current_score
            
    return best if best_score >= 1 else ""

def _build_chroma_db(df_clean: pd.DataFrame):
    global _chroma_collection
    print("[DATA] Building ChromaDB collection for flexible rate lookup...")
    
    client = chromadb.Client() 
    _chroma_collection = client.get_or_create_collection("price_transparency_rates")

    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df_clean.iterrows():
        h = row['hospital_name']
        c = row['code']
        p = row['payer_name']
        l = row['plan_name']
        r = row['rate']

        desc = row['description'] if pd.notna(row['description']) else "No description"
        doc_text = f"The price for CPT {c} ({desc}) at {h} for payer {p} with plan {l} is ${r:,.2f}."
        
        metadata = {
            'cpt': c,
            'hospital': h,
            'payer': p,
            'plan': l,
            'rate': r
        }
        
        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(f"doc_{idx}")
        
    try:
        # Add data to ChromaDB
        _chroma_collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"[DATA] ChromaDB populated with {len(ids)} rate documents.")
    except Exception as e:
        print(f"[CHROMA ERROR] Failed to populate ChromaDB. Proceeding without Tier 0. Error: {e}")


def retrieve_rates_from_chroma(weighted_codes, payer, plan, hospitals, collection):
    if not collection:
        return {}
    cpt_list = ", ".join([f"{c} (weight: {w:.2f})" for c, w in weighted_codes])
    hospital_list = ", ".join(hospitals)
    
    query_text = (
        f"Find negotiated rates for the top procedure codes: {cpt_list}. "
        f"The procedure intent is '{weighted_codes[0][0]}'. "
        f"The relevant hospitals are: {hospital_list}. "
        f"The patient's insurance is {payer} with plan {plan}."
    )
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=50,
        )
    except Exception as e:
        print(f"[CHROMA QUERY ERROR] Failed to query ChromaDB. Error: {e}")
        return {}
    
    chroma_rates = {} # Key: hospital -> cpt -> List[rate]
    
    if results and results.get('metadatas'):
        for meta in results['metadatas'][0]:
            # Chroma returns metadata dictionaries
            h = meta.get('hospital')
            c = meta.get('cpt')
            r = meta.get('rate')
            
            if h and c and r is not None:
                if h not in chroma_rates:
                    chroma_rates[h] = {}
                if c not in chroma_rates[h]:
                    chroma_rates[h][c] = []
                chroma_rates[h][c].append(r)
                
    return chroma_rates


def _build_df_and_dicts(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The required data file was not found at: {path}. Please check your CSV_PATH.")
        
    print(f"[DATA] Reading CSV into DataFrame: {path}...")
    
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines='skip', low_memory=False)
    
    df['rate'] = df.apply(get_rate, axis=1)
    df_clean = df.dropna(subset=['rate']).copy()
    
    df_clean['hospital_name'] = df_clean['hospital_name'].astype(str).str.strip()
    df_clean['payer_name'] = df_clean['payer_name'].astype(str).str.strip().replace('', np.nan)
    df_clean['plan_name'] = df_clean['plan_name'].astype(str).str.strip().replace('', np.nan)
    df_clean['code'] = df_clean['code'].astype(str).str.strip()
    df_clean['description'] = df_clean['description'].astype(str).str.strip()
    
    hospitals = df_clean['hospital_name'].unique().tolist()
    payers = df_clean['payer_name'].dropna().unique().tolist()
    plans = df_clean['plan_name'].dropna().unique().tolist()
    
    code2desc = df_clean.drop_duplicates(subset=['code'], keep='first').set_index('code')['description'].to_dict()
    
    print("\n[DATA] Building Rate Lookup Dictionaries (rows_plan, rows_all)...")
    rows_plan = {} 
    rows_all = {}  

    for _, row in df_clean.iterrows():
        h = row['hospital_name']
        c = row['code']
        r = row['rate']
        p = row['payer_name']
        l = row['plan_name']
        
        key_all = (h, c)
        if key_all not in rows_all:
            rows_all[key_all] = []
        rows_all[key_all].append(r)
        
        if pd.notna(p):
            plan_key = l if pd.notna(l) else None 
            key_plan = (h, c, p, plan_key) 
            if key_plan not in rows_plan:
                rows_plan[key_plan] = []
            rows_plan[key_plan].append(r)
            
    print("[DATA] Lookup dictionaries built.")

    return (
        hospitals, payers, plans, 
        code2desc, df_clean, 
        rows_plan, rows_all
    )

def build_vocab(path):
    hospitals, payers, plans, code2desc, df_clean, rows_plan, rows_all = _build_df_and_dicts(path)
    
    print("\n[DATA] Calculating CPT description embeddings...")
    codes_in_order = list(code2desc.keys()) 
    descriptions_in_order = [code2desc[c] for c in codes_in_order]
    
    description_embeddings = retrieval_model.encode(
        descriptions_in_order, 
        convert_to_tensor=True,
        batch_size=64 
    ).cpu().numpy() 
    print("[DATA] Embedding calculation complete.")
    
    if not df_clean.empty:
        _build_chroma_db(df_clean)

    return (
        hospitals, payers, plans, 
        code2desc, df_clean, 
        codes_in_order, description_embeddings,
        rows_plan, rows_all
    )

def semantic_cpt_candidates(query: str, codes_in_order: List[str], description_embeddings: np.ndarray, k: int = 12):
    """Finds the top K CPT codes semantically closest to the user's query."""
    if not query.strip(): return []

    query_embedding = retrieval_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    # Cosine Similarity Calculation
    cos_scores = np.dot(query_embedding, description_embeddings.T)[0]
    top_indices = np.argsort(cos_scores)[::-1][:k]
    cpt_candidates = [codes_in_order[idx] for idx in top_indices]
    
    return cpt_candidates


def llm_extract_slots(user_text: str, hospitals: List[str], payers: List[str], plans: List[str], k_hospitals=4):
    """Uses LLM to extract key entities and snaps them to the vocabulary."""
    gen = _build_llm()
    prompt = f"""
<<SYS>>
You are an expert at extracting clinical and financial context from patient requests.
You must return ONLY a compact JSON object. Do not include any explanation or markdown before or after the JSON.
<<SYS>>

Extract the following fields from the patient's text:
1. 'payer': The insurance provider's name.
2. 'plan': The specific plan type or name (if provided).
3. 'hospitals': A list of up to 3 specific hospital names or health systems mentioned.
4. 'intent': A very brief (5-10 word) summary of the clinical procedure, symptom, or treatment requested.

JSON structure:
{{
  "payer": "<insurer or empty>",
  "plan": "<plan type or name or empty>",
  "hospitals": ["<hospital/system/area>", ... up to 3],
  "intent": "<very brief clinical intent>"
}}

Text:
{user_text}

JSON:
"""
    out = gen(prompt, max_new_tokens=220)
    m = re.search(r"\{.*\}", out, flags=re.S)
    
    payer = plan = intent = ""
    hospitals_raw = []

    if m:
        try:
            js = json.loads(m.group(0))
            payer = str(js.get("payer","")).strip()
            plan  = str(js.get("plan","")).strip()
            intent= str(js.get("intent","")).strip()
            hospitals_raw = js.get("hospitals") or []
            if not isinstance(hospitals_raw, list): hospitals_raw = []
        except: pass

    payer = _best_match(payer, payers) if payer else ""
    plan  = _best_match(plan, plans) if plan else ""

    snapped = []
    for h in hospitals_raw:
        snap = _best_match(h, hospitals)
        if snap and snap not in snapped:
            snapped.append(snap)
        if len(snapped) >= k_hospitals: break

    return payer, plan, snapped, intent

def llm_weight_codes(intent_text: str, candidate_codes: List[str], code2desc: Dict[str,str], top_k: int = 5) -> List[Tuple[str, float]]:
    """Uses LLM to rank and weight the most relevant CPT codes."""
    gen = _build_llm()
    subset = candidate_codes[:max(top_k*2, top_k)]
    items = [{"code": c, "desc": code2desc.get(c,"")} for c in subset]

    prompt = (
        f"""
<<SYS>>
You are an expert medical coder. You must return ONLY a compact JSON object.
The JSON object must map the CPT code (string key) to its relevance weight (float value) for the patient's intent. 
The weights must sum to approximately 1.0. Pick at most 5 codes.
Crucially, **DISREGARD ALL SUPPLY, MATERIAL, OR LOW-VALUE CODES** (e.g., Q codes, A codes, non-procedure/service codes).
Focus only on the **main clinical procedure code(s)** relevant to the patient's intent.
<<SYS>>

Pick and weight the 5 most relevant CPT codes for this patient intent.
Return ONLY JSON object mapping CPT code to weight (float value).

Intent:
{intent_text}

Candidates (code, description):
{json.dumps(items, ensure_ascii=False)}

JSON:
"""
    )
    
    out = gen(prompt, max_new_tokens=220)
    m = re.search(r"\{.*\}", out, flags=re.S)
    
    if not m:
        picks = subset[:top_k]
        w = 1.0 / max(1,len(picks))
        return [(c, w) for c in picks]
        
    try:
        weights = json.loads(m.group(0))
        
        filtered = []
        for c in subset:
            try:
                w = float(weights.get(c, 0.0))
            except:
                w = 0.0
                
            if w > 0: filtered.append((c, w))
            
        filtered.sort(key=lambda x: -x[1])
        picks = filtered[:top_k] if len(filtered) >= top_k else filtered
        
        s = sum(w for _,w in picks) or 1.0
        return [(c, w/s) for c,w in picks]
        
    except:
        picks = subset[:top_k]
        w = 1.0 / max(1,len(picks))
        return [(c, w) for c in picks]


#main program

def main():
    global _chroma_collection
    print(f"\n--- Price Transparency RAG Tool ---")
    
    try:
        (
            hospitals, payers, plans, code2desc, df_clean, 
            codes_in_order, description_embeddings, 
            rows_plan, rows_all
        ) = build_vocab(CSV_PATH)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        return
    except Exception as e:
        print(f"FATAL ERROR during data loading/embedding: {e}")
        return

    if df_clean.empty:
        print("Error: No valid rate data found in the CSV after cleaning. Cannot proceed.")
        return

    patient_text = ask("Tell me your situation (symptoms/treatment) and insurance, in your own words (e.g., 'I need a knee replacement at UPMC Carlisle' or 'a colonoscopy near Pittsburgh').")

    print("\n[STEP 1/6: LLM Slot Extraction...]")
    payer, plan, picked_hospitals, intent = llm_extract_slots(patient_text, hospitals, payers, plans, k_hospitals=4)

    if not picked_hospitals:
        extra = ask("Which specific hospital(s) or area are you interested in? (e.g., 'UPMC near Pittsburgh').")
        
        _, _, snapped_hospitals_extra, _ = llm_extract_slots(extra, hospitals, payers, plans, k_hospitals=4)
        
        if snapped_hospitals_extra: 
            picked_hospitals = snapped_hospitals_extra
        else:
            print("Warning: No specific hospital found, defaulting to top 4 in the dataset.")
            picked_hospitals = hospitals[:4]

    # Semantic Search: Find CPT candidates by intent
    print("\n[STEP 2/6: Semantic CPT Candidate Retrieval...]")
    intent_for_codes = intent or patient_text
    cpt_candidates = semantic_cpt_candidates(
        intent_for_codes, 
        codes_in_order, 
        description_embeddings, 
        k=20 
    )
    if not cpt_candidates:
        print("Warning: Semantic search failed, defaulting to first 20 codes in the dataset.")
        cpt_candidates = list(code2desc.keys())[:20]

    #LLM Re-ranking
    print("[STEP 3/6: LLM CPT Re-ranking/Weighting...]")
    weighted_codes = llm_weight_codes(intent_for_codes, cpt_candidates, code2desc, top_k=5)
    
    print("[STEP 4/6: Prioritizing Payer Lookups...]")
    
    allpairs = list({(k[2], k[3]) for k in rows_plan.keys()}) 
    ph, pl = (payer or "").lower(), (plan or "").lower()
    
    def pair_rank(x):
        p_ok = ph and ph in (x[0] or "").lower()
        l_ok = pl and pl in (x[1] or "").lower()
        return (not (p_ok and l_ok), not p_ok, not l_ok)
        
    ranked_pairs = sorted(allpairs, key=pair_rank)

    print("[STEP 5/6: Calculating Weighted Median Cost with Confidence Tracking...]")
    results = []
    
    rows_ultimate_fallback_median = {} 
    for code in df_clean['code'].unique():
        rates = df_clean[df_clean['code'] == code]['rate'].tolist()
        rows_ultimate_fallback_median[code] = median(rates)
        
    chroma_rates = retrieve_rates_from_chroma(weighted_codes, payer, plan, picked_hospitals, _chroma_collection)

    for h in picked_hospitals[:4]:
        total_w = tw = 0.0
        max_tier_used = 0 
        
        for code, w in weighted_codes:
            median_rate = None
            current_tier = 0

            if h in chroma_rates and code in chroma_rates[h]:
                vals = chroma_rates[h][code]
                if vals:
                    median_rate = median(vals)
                    current_tier = 0.5 
            
            if pd.isna(median_rate):
                for payr, pln in ranked_pairs:
                    plan_key = pln if pd.notna(pln) else None 
                    vals = rows_plan.get((h, code, payr, plan_key))
                    
                    if vals:
                        median_rate = median(vals)
                        if pd.notna(median_rate):
                            current_tier = 1
                            break

            if pd.isna(median_rate):
                vals = rows_all.get((h, code)) 
                if vals:
                    median_rate = median(vals)
                    if pd.notna(median_rate):
                        current_tier = 2
            
            if pd.isna(median_rate):
                median_rate = rows_ultimate_fallback_median.get(code)
                if pd.notna(median_rate):
                    current_tier = 3
                        
            if pd.notna(median_rate) and median_rate > 0:
                total_w += (w * median_rate)
                tw += w
                
                if max_tier_used == 0 or current_tier < max_tier_used:
                     max_tier_used = current_tier
                
        est = (total_w / tw) if tw > 0 else None
        results.append((h, est, str(max_tier_used) if max_tier_used == 0.5 else int(max_tier_used))) 

    print("\n[STEP 6/6: Generating Patient-Friendly Explanation]")
    print("\n--- Estimated Cost and Explanation ---\n")
    if intent:
        print(f"Based on your description ('{intent}'), the system identified the 5 most relevant CPT billing codes for your expected procedure:")
    else:
        print("Based on your description, the system identified the 5 most relevant CPT billing codes for your expected procedure:")

    print("\n## 1. Top 5 Estimated CPTs and Relevance:")
    for code, w in weighted_codes:
        desc = code2desc.get(code, "Description not found.")
        short = (desc[:80] + "…") if len(desc) > 80 else desc
        print(f"  • **{code}**: {short} (Relevance Weight: {w:.2f})")

    if payer or plan:
        print(f"\n## 2. Insurance Context Used:")
        print(f"  • **Payer**: {payer or 'Not found'} | **Plan**: {plan or 'Not found'}")
        print("The estimates below prioritize the negotiated rates found for this Payer/Plan combination.")
    else:
        print("\n## 2. Insurance Context:")
        print("No specific payer or plan was found, so estimates are based on the median negotiated rate.")
        
    print("\n## 3. Estimated Weighted Total Cost and Data Confidence:")
    print("(This is a weighted median of the 5 CPTs above, based on their relevance to your query.)")
    
    tier_map = {
        0.5: "Tier 0: High Flexibility/Medium Confidence (RAG Search)",
        1: "Tier 1: High Confidence (Specific Payer Negotiated Rate)",
        2: "Tier 2: Medium Confidence (Hospital Median Rate - Any Payer)",
        3: "Tier 3: Low Confidence (System Median Rate - All Hospitals/All Payers)",
        0: "Data Insufficient."
    }
    
    has_estimate = False
    for h, est, tier in results:
        tier_text = tier_map.get(tier, "Error in Confidence Tracking.")
        if est is not None:
            print(f"  • **{h}**: ${est:,.2f} ({tier_text})")
            has_estimate = True
        else:
            print(f"  • **{h}**: {tier_map.get(0)}")

    if not has_estimate:
        print("\n**Important Note**: We could not find reliable cost data for the relevant codes across the entire dataset.")

    print("\n--- Explanation of Cost Derivation ---")
    formula = r"$$\text{Estimated Cost} = \sum_{i=1}^{5} (W_i \times M_i)$$"
    explanation = (
        r"Where:"
        r"\n* $\text{i}$ is one of the top 5 CPT codes."
        r"\n* $\text{W}_i$ is the Relevance Weight (likelihood) determined by Llama-2 (Step 3)."
        r"\n* $\text{M}_i$ is the Payer-Specific Median Rate for CPT $\text{i}$."
    )

    display(Markdown(
        "The final estimated cost for a hospital is calculated as the weighted sum of the 5 most relevant CPT codes:"
        + formula
        + explanation
    ))

    print(f"\n#### How the Median Rate (M_i) is Determined (Four-Tiered Fallback):")
    print(f"The system uses a strict hierarchical lookup (a four-tiered data retrieval logic) to find the most reliable rate for each CPT code: ")
    
    print("* **Tier 0 (Highest Flexibility):** Attempts a **flexible RAG search** using ChromaDB to find similar rates based on the entire patient context (hospital, payer, CPT intent).")
    print("* **Tier 1 (Highest Confidence):** If Tier 0 is empty, searches for the negotiated rate specific to your **Payer** and **Plan** at the **Hospital** (Rigid Match).")
    print("* **Tier 2 (Medium Confidence):** If Tier 1 fails, falls back to the overall **Median Rate** for that CPT code at the specific **Hospital** (ignoring Payer/Plan).")
    print("* **Tier 3 (Lowest Confidence):** If Tier 2 fails, uses the **System-Wide Median Rate** for that CPT code, calculated across **All Hospitals** and All Payers.")
    
    print("\nThis RAG system minimizes hallucination by grounding all cost facts directly in the price transparency data, using the highest available confidence tier.")

if __name__ == "__main__":
    main()
