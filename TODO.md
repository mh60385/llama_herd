# TODO: Research-Grade Experiment Roadmap

This file tracks all outstanding work to bring the experiment to publication-ready quality.

---

## 🎯 **CRITICAL: Must Fix Before Publication**

### 1. Embedding-Based Semantic Drift Metrics
- **File:** `src/drift_analysis.py`
- **Status:** ⚠️ Placeholder (Jaccard) implemented, embeddings required for publication
- **Action:** Replace `_jaccard_similarity()` with sentence-transformers
- **Blocker:** Jaccard similarity does not capture semantic meaning
- **Code:**
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  ```
- **Dependency:** Add `sentence-transformers>=2.2.0` to requirements.txt
- **Testing:** Verify drift scores correlate with human judgment

---

## 📊 **HIGH PRIORITY: Research Rigor**

### 2. Define Formal Hypotheses
- **File:** `RESEARCH_README.md` (new section)
- **Status:** ❌ Missing
- **Action:** Add Hypotheses section with testable claims
- **Example hypotheses:**
  - H1: Anti-anchoring threshold of 0.50 reduces query repetition rate by ≥30% compared to 0.72
  - H2: Temperature 1.0 produces higher semantic diversity (drift > 0.7) than temperature 0.7
  - H3: Relaxed interest thresholds (2+ episodes, 1+ domain) promote interests faster than strict (3+, 2+)
  - H4: LLM agents show higher drift scores than random baseline
- **Metric:** Define primary outcome measure for each hypothesis

### 3. Power Analysis & Sample Size Justification
- **File:** `RESEARCH_README.md` (new section)
- **Status:** ❌ Missing
- **Action:** Calculate required sample size
- **Current:** 5 agents × 100 episodes = 500 episodes
- **Justification needed:**
  - What effect size can we detect with 80% power at p<0.05?
  - Is 100 episodes per agent sufficient?
  - Should we run more agents or more episodes per agent?
- **Tool:** Use G*Power or statistical power calculation

### 4. Control Groups for Experimental Design
- **File:** New experiment configuration
- **Status:** ❌ Missing
- **Action:** Run controlled comparisons
- **Conditions to compare:**
  | Group | Anti-anchoring | Temperature | Thresholds | N |
  |-------|----------------|-------------|------------|---|
  | Control | 0.72 | 0.7 | 3+, 2+ | 5 agents × 100 eps |
  | Experimental | 0.50 | 1.0 | 2+, 1+ | 5 agents × 100 eps |
- **Analysis:** Compare drift scores, repetition rates, interest promotion speed

### 5. Statistical Analysis
- **File:** New `scripts/statistical_analysis.py`
- **Status:** ❌ Missing
- **Action:** Implement proper statistical tests
- **Tests needed:**
  - Paired t-tests: Compare LLM vs baseline drift scores
  - ANOVA: Compare multiple groups (control vs experimental)
  - Effect sizes: Cohen's d for standardized mean differences
  - Confidence intervals: 95% CI for all reported metrics
- **Output:** p-values, effect sizes, confidence intervals
- **Example:**
  ```python
  from scipy import stats
  t_stat, p_value = stats.ttest_rel(baseline_drifts, agent_drifts)
  # Report: t(48) = 2.45, p = 0.018, d = 0.67
  ```

---

## 📈 **MEDIUM PRIORITY: Enhanced Analysis**

### 6. Visualization
- **File:** New `scripts/visualize_drift.py`
- **Status:** ❌ Missing
- **Charts needed:**
  - Drift over time (line chart per agent)
  - Distribution comparison (box plot: LLM vs baselines)
  - Drift trend per agent (bar chart)
  - Interest evolution (Sankey diagram of interest promotion)
- **Output:** PNG/SVG files in `data/metrics/figures/`

### 7. Interest Promotion Analysis
- **File:** Enhance `analyze()` or new script
- **Status:** ❌ Missing
- **Metrics:**
  - Time to promotion (episodes until interest becomes stable)
  - Promotion rate (interests promoted per episode)
  - Rejection rate (interests rejected per episode)
  - Source diversity per interest (domains supporting each interest)

### 8. Error Analysis
- **File:** Enhance `analyze()`
- **Status:** ⚠️ Partial
- **Action:** Categorize and analyze errors systematically
- **Categories:**
  - Source reader failures (403, timeout, etc.)
  - Profile update rejections (by reason)
  - Interest rejections (by reason)
  - Model output errors (JSON parsing, etc.)
- **Output:** Error rate by category, trends over time

---

## 🧹 **LOW PRIORITY: Cleanup & Polish**

### 9. Backend Simplification
- **File:** `.env`, `configs/experiment.yaml`, `src/search.py`
- **Status:** ⚠️ Optional backends enabled
- **Action:** Default to SearXNG only for reproducibility
- **Current state:**
  - ✅ Wikipedia backend implemented
  - ✅ GDELT backend implemented
  - ✅ Configuration exists
  - ⚠️ Both disabled by default in `.env`
- **Recommendation:** Keep code, disable by default, enable for specific experiments

### 10. Malformed Interest Handling
- **File:** `src/utils.py` (BLOCKED_PROFILE_TERMS, etc.)
- **Status:** ⚠️ Issues seen in logs
- **Problem:** Interests with quotes: `"Gender"`, `"Non-fiction"]`
- **Action:** Add validation to `_safe_interest()` or sanitize model output
- **Example:** Strip quotes, check for malformed JSON fragments

### 11. GDELT API Error Handling
- **File:** `src/search.py` (_search_gdelt)
- **Status:** ⚠️ Rate-limited/errors seen
- **Problem:** 429 Too Many Requests, JSON parsing errors
- **Action:** Add retry logic with exponential backoff
- **Code:**
  ```python
  import time
  max_retries = 3
  for attempt in range(max_retries):
      try:
          response = requests.get(...)
          return results
      except (RequestException, JSONDecodeError):
          if attempt < max_retries - 1:
              time.sleep(2 ** attempt)
  ```

### 12. Source 403 Error Handling
- **File:** `src/source_reader.py`
- **Status:** ⚠️ Bloomsbury 403 errors
- **Problem:** Commercial sites block scraping
- **Action:** 
  - Add User-Agent header rotation
  - Implement fallback to cached content
  - Exclude known blocking domains

---

## 📚 **Documentation Updates**

### 13. Update Methodology Section
- **File:** `RESEARCH_README.md`
- **Status:** ⚠️ Needs expansion
- **Add:**
  - Hypotheses (see #2)
  - Experimental design (see #4)
  - Statistical methods (see #5)
  - Sample size justification (see #3)
  - Limitations section

### 14. Add Architecture Diagram
- **File:** New `docs/architecture.md` or `ARCHITECTURE.md`
- **Status:** ❌ Missing
- **Content:**
  ```
  ┌─────────────────────────────────────────────┐
  │                 Episode Runner                │
  ├─────────────────────────────────────────────┤
  │ 1. Load Profile ─────► 2. Generate Query     │
  │                    (temp=1.0, anti-anchor=0.50)│
  │ 3. Search ─────────► 4. Select Source      │
  │    (SearXNG)          (LLM)                   │
  │ 5. Summarize ───────► 6. Diary Entry        │
  │    (LLM)               (LLM)                  │
  │ 7. Profile Update ┌────────────────────────┐│
  │    (thresholds: 2+,  │   8. Save Episode      ││
  │     domains: 1+)     │    (JSONL + SQLite)   ││
  └─────────────────────┴─────────────────────┘
  ```

### 15. Add Data Dictionary
- **File:** New `docs/data_dictionary.md`
- **Status:** ❌ Missing
- **Content:** Document all data files and their schemas
- **Example:**
  ```markdown
  ### data/profiles/agent_01.json
  - agent_id: string
  - current_interests: list[string]
  - stable_interests: list[string]
  - version: int
  - observations: list[dict]
  
  ### data/logs/episodes.jsonl
  - episode_id: string
  - agent_id: string
  - search_query: string
  - search_results: list[SearchResult]
  - diary_entry: DiaryEntry
  - reflection: Reflection (removed in v2)
  ```

---

## ✅ **COMPLETED**

### Code Changes
- ✅ Anti-anchoring threshold: 0.72 → 0.50
- ✅ Wikipedia backend: Implemented and tested
- ✅ Interest thresholds: (3+, 2+) → (2+, 1+)
- ✅ Wikipedia seeding: Categorized by domain
- ✅ Reflection: Removed for efficiency
- ✅ Temperature: 0.7 → 1.0
- ✅ Semantic drift: Jaccard placeholder implemented

### Documentation
- ✅ PIPELINE_SUMMARY.md: Updated with baselines, backends, semantic drift
- ✅ README.md: Updated assumptions, drift analysis note
- ✅ RESEARCH_README.md: Updated episode flow, drift interpretation

### Baselines
- ✅ `src/baselines.py`: 4 baseline types (random, static, echo, repeating)
- ✅ `scripts/compare_baselines.py`: Comparison script
- ✅ Documentation: Baseline table in PIPELINE_SUMMARY.md

---

## 📅 **Recommended Work Order**

### Phase 1: Critical (Publication Blockers)
1. ✅ ~~Implement baselines~~ **DONE**
2. **Upgrade to embedding-based semantic drift** (Highest priority)
3. Define formal hypotheses
4. Add control groups

### Phase 2: Analysis Rigor
5. Add statistical testing
6. Add visualizations
7. Power analysis

### Phase 3: Polish
8. Error handling (GDELT, 403s)
9. Malformed interest cleanup
10. Backend simplification
11. Documentation (architecture, data dictionary)

---

## 🎯 **Minimal Viable for Next Paper Draft**

To have a paper-ready experiment, complete:
- [ ] **#2 Embedding upgrade** (non-negotiable)
- [ ] **#3 Hypotheses** (non-negotiable)
- [ ] **#4 Control groups** (non-negotiable)
- [ ] **#5 Statistical testing** (highly recommended)

**Estimated effort:** 2-3 days of focused work

---

## 📌 **Notes**

### On Semantic Drift Placeholder
The current Jaccard similarity implementation is **NOT suitable for publication**. 
It provides a useful placeholder for development but must be replaced with embeddings.

### On Baseline Agents
Baselines are implemented and tested. They provide a foundation for comparing 
LLM agent behavior against simple alternatives. This is essential for claiming 
that LLM agents are "better" than random.

### On Backend Simplification
For reproducibility, **use SearXNG only** (web search). Wikipedia and GDELT 
can be enabled for specific experiments but should be disabled by default.

### On Current Results
Agent_01 drift score (0.706) is below random baseline (0.990). This suggests:
- The agent maintains topic focus (good for coherence)
- May not be exploring enough (needs investigation)
- The baselines help quantify this trade-off

---

*Last updated: 2026-05-30*
