#!/usr/bin/env python3
"""VEP AI Assistant — recommends Ensembl VEP configuration based on your analysis scenario.

Supports three modes:
  python vep_assistant.py                        # interactive recommendation
  python vep_assistant.py --explain "query"      # recommendation + decision trace
  python vep_assistant.py explain-result "why..." # explain a VEP output annotation
"""

import json
import os
import re
import sys
import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai SDK not installed. Run: pip install openai")
    sys.exit(1)

BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Knowledge base loading
# ---------------------------------------------------------------------------

def load_knowledge_base():
    """Load VEP options and training examples from JSON files."""
    options_path = BASE_DIR / "vep_options.json"
    examples_path = BASE_DIR / "training_examples.json"

    if not options_path.exists():
        print(f"Error: VEP options file not found at {options_path}")
        sys.exit(1)
    if not examples_path.exists():
        print(f"Error: Training examples file not found at {examples_path}")
        sys.exit(1)

    with open(options_path) as f:
        vep_options = json.load(f)
    with open(examples_path) as f:
        training_examples = json.load(f)

    return vep_options, training_examples


def load_consequences():
    """Load VEP consequence term definitions."""
    path = BASE_DIR / "vep_consequences.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fuzzy option extraction from free-text LLM output
# ---------------------------------------------------------------------------

def build_option_aliases(vep_options):
    """Build a map of alias → option_id for fuzzy matching."""
    aliases = {}
    for opt in vep_options:
        oid = opt["id"]
        # canonical id
        aliases[oid.lower()] = oid
        # name
        aliases[opt["name"].lower()] = oid
        # cli flags (split on /)
        for flag in re.split(r"[/,\s]+", opt["cli_flag"]):
            flag = flag.strip().lstrip("-").lower()
            if len(flag) > 2:
                aliases[flag] = oid
    # common extra aliases
    extras = {
        "polyphen2": "polyphen", "polyphen-2": "polyphen",
        "splice_ai": "spliceai", "splice ai": "spliceai",
        "alpha_missense": "alphamissense", "alpha missense": "alphamissense",
        "gnomad": "gnomad_af", "gnomad_freq": "gnomad_af",
        "gnomad_sv_freq": "gnomad_sv",
        "1000genomes": "af_1kg", "1000_genomes": "af_1kg", "1kg": "af_1kg",
        "af_1kg": "af_1kg",
        "maxentscan": "maxentscan", "max_ent_scan": "maxentscan",
        "mane": "mane_select",
        "gene_pheno": "gene_phenotype", "phenotype": "gene_phenotype",
        "existing": "check_existing", "check existing": "check_existing",
        "clinvar_structural": "clinvar_sv",
        "gnomad_structural": "gnomad_sv",
    }
    for alias, oid in extras.items():
        aliases[alias] = oid
    return aliases


def _match_option(text, aliases):
    """Try to match a text fragment to an option id.

    Uses direct matching first, then substring matching with a minimum
    length of 4 characters to avoid false positives from short fragments.
    """
    text = text.strip().lower().replace("-", "_").replace(" ", "_")
    # direct
    if text in aliases:
        return aliases[text]
    # strip leading dashes (cli flags)
    stripped = text.lstrip("_")
    if stripped in aliases:
        return aliases[stripped]
    # substring match — require both sides >= 4 chars to reduce false positives
    if len(text) >= 4:
        for alias, oid in sorted(aliases.items(), key=lambda x: -len(x[0])):
            if len(alias) >= 4 and (alias in text or text in alias):
                return oid
    return None


def extract_recommendations(text, option_aliases):
    """Parse LLM output to extract which options are enabled/disabled.

    Parses table rows first, then also scans non-table prose lines so that
    options mentioned outside tables are not silently dropped.  Prose
    scanning uses word-boundary matching to avoid false positives from
    common English words that happen to match short alias strings.
    """
    enabled = set()
    disabled = set()
    text_lower = text.lower()

    # --- Phase 1: parse table rows ---
    table_rows = re.findall(
        r"\|\s*\*{0,2}([^|]+?)\*{0,2}\s*\|\s*\*{0,2}(enable|disable|on|off|yes|no|true|false)\*{0,2}\s*\|",
        text_lower,
    )
    for opt_text, status in table_rows:
        opt_text = opt_text.strip().strip("`").strip("*")
        matched = _match_option(opt_text, option_aliases)
        if matched:
            if status in ("enable", "on", "yes", "true"):
                enabled.add(matched)
            else:
                disabled.add(matched)

    # --- Phase 2: scan non-table lines for prose recommendations ---
    for line in text_lower.split("\n"):
        # Skip table rows (already parsed above)
        if "|" in line:
            continue
        for alias, oid in option_aliases.items():
            if oid in enabled or oid in disabled:
                continue  # already resolved via table
            # Word-boundary matching to avoid "pick" matching "picking" etc.
            if not re.search(r"\b" + re.escape(alias) + r"\b", line):
                continue
            # check context around the alias
            if re.search(r"(enabl|turn.{0,3}on|\bon\b|recommend|include|add|use )", line):
                enabled.add(oid)
            elif re.search(r"(disabl|turn.{0,3}off|\boff\b|skip|omit|not.{0,6}need|unnecessary|don.t)", line):
                disabled.add(oid)

    return enabled, disabled


# ---------------------------------------------------------------------------
# Post-hoc constraint checker (runs AFTER LLM output, BEFORE display)
# ---------------------------------------------------------------------------

# Priority ranking for conflict resolution (higher number = higher priority)
_PRIORITY_RANK = {
    "critical": 4,
    "recommended": 3,
    "optional": 2,
    "not_applicable": 1,
}

# Restrictiveness ranking: when priorities are equal, disable the MORE restrictive
# option first (most_severe is most restrictive because it suppresses annotations)
_RESTRICTIVENESS = {
    "most_severe": 3,
    "pick": 2,
    "per_gene": 1,
}

# Keyword → species mapping for species inference
_SPECIES_KEYWORDS = {
    "mouse": "mouse",
    "mus musculus": "mouse",
    "grcm": "mouse",
    "grcm38": "mouse",
    "grcm39": "mouse",
    "zebrafish": "zebrafish",
    "danio": "zebrafish",
    "danio rerio": "zebrafish",
    "drosophila": "drosophila",
    "fruit fly": "drosophila",
    "d. melanogaster": "drosophila",
    "c. elegans": "c_elegans",
    "caenorhabditis": "c_elegans",
    "rat": "rat",
    "rattus": "rat",
    "chicken": "chicken",
    "gallus": "chicken",
    "pig": "pig",
    "sus scrofa": "pig",
    "dog": "dog",
    "canis": "dog",
    "non-human": "non_human",
    "non human": "non_human",
    "arabidopsis": "arabidopsis",
    "rice": "rice",
    "oryza": "rice",
}


def infer_species(user_query: str) -> str:
    """Detect species from the user query using word-boundary matching.

    Returns the species name (e.g. 'mouse', 'zebrafish') or 'human' (default).
    Uses regex word boundaries to avoid false positives like 'rat' in 'generated'.
    """
    query_lower = user_query.lower()
    for keyword, species in _SPECIES_KEYWORDS.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
            return species
    return "human"


def _get_priority_rank(option_id: str, use_case: str, vep_options: list) -> int:
    """Look up the numeric priority rank for an option in a given use case."""
    for opt in vep_options:
        if opt["id"] == option_id:
            priority = opt.get("priority_by_use_case", {}).get(use_case, "not_applicable")
            return _PRIORITY_RANK.get(priority, 0)
    return 0


def _detect_use_case(enabled: set, vep_options: list, training_examples: list,
                     user_query: str) -> str:
    """Infer the use case category from the top retrieval match."""
    scored = []
    query_words = set(user_query.lower().split())
    for ex in training_examples:
        ex_text = f"{ex['user_query']} {ex['use_case_category']} {ex.get('justification', '')}".lower()
        ex_words = set(ex_text.split())
        overlap = len(query_words & ex_words)
        scored.append((overlap, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]["use_case_category"] if scored else "rare_disease_germline"


def check_and_fix_violations(enabled: set, disabled: set, vep_options: list,
                             training_examples: list,
                             user_query: str) -> list[dict]:
    """Check enabled options for constraint violations and auto-correct them.

    Loads conflict rules and species restrictions from vep_options.json.
    For conflicts, disables the option with lower priority for the detected
    use case. If priorities are equal, disables the more restrictive option
    (most_severe > pick > per_gene).

    Returns a list of violation dicts with keys:
        type: 'conflict' or 'species'
        option_disabled: the option that was disabled
        option_kept: (conflicts only) the option that was kept
        reason: human-readable explanation
    """
    violations = []
    species = infer_species(user_query)
    use_case = _detect_use_case(enabled, vep_options, training_examples, user_query)

    # Build lookup maps
    conflicts_map = {}
    species_map = {}
    description_map = {}
    for opt in vep_options:
        conflicts_map[opt["id"]] = set(opt.get("conflicts_with", []))
        species_map[opt["id"]] = opt.get("species_restriction", "all species")
        description_map[opt["id"]] = opt.get("description", "")[:80]

    # --- Species violations ---
    if species != "human":
        for oid in list(enabled):
            restriction = species_map.get(oid, "all species").lower()
            # Flag as human-only if restriction mentions "human" but NOT
            # "all species" and NOT "human and" (which implies multi-species)
            is_human_only = (
                "human" in restriction
                and "all" not in restriction
                and "human and" not in restriction
            )
            if is_human_only:
                violations.append({
                    "type": "species",
                    "option_disabled": oid,
                    "reason": (
                        f"'{oid}' is restricted to {species_map[oid]} "
                        f"but your query specifies {species}"
                    ),
                })
                enabled.discard(oid)
                disabled.add(oid)

    # --- Conflict violations ---
    checked_pairs = set()
    for oid_a in list(enabled):
        for oid_b in list(enabled):
            if oid_a == oid_b:
                continue
            pair = tuple(sorted([oid_a, oid_b]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            if oid_b in conflicts_map.get(oid_a, set()) or oid_a in conflicts_map.get(oid_b, set()):
                # Decide which to disable: lower priority loses
                rank_a = _get_priority_rank(oid_a, use_case, vep_options)
                rank_b = _get_priority_rank(oid_b, use_case, vep_options)

                if rank_a != rank_b:
                    loser = oid_a if rank_a < rank_b else oid_b
                    winner = oid_b if loser == oid_a else oid_a
                else:
                    # Equal priority: disable the more restrictive option
                    rest_a = _RESTRICTIVENESS.get(oid_a, 0)
                    rest_b = _RESTRICTIVENESS.get(oid_b, 0)
                    if rest_a != rest_b:
                        loser = oid_a if rest_a > rest_b else oid_b
                        winner = oid_b if loser == oid_a else oid_a
                    else:
                        # Fallback: disable the first alphabetically
                        loser, winner = sorted([oid_a, oid_b])

                # Find the conflict reason from whichever side declared it
                if loser in conflicts_map.get(winner, set()):
                    decl = winner
                else:
                    decl = loser
                conflict_note = (
                    f"--{decl} conflicts with --{loser}" if decl != loser
                    else f"--{loser} conflicts with --{winner}"
                )

                violations.append({
                    "type": "conflict",
                    "option_disabled": loser,
                    "option_kept": winner,
                    "reason": (
                        f"'{loser}' and '{winner}' cannot both be enabled "
                        f"({conflict_note}). Disabled: {loser}"
                    ),
                })
                enabled.discard(loser)
                disabled.add(loser)

    return violations


def format_violation_warnings(violations: list[dict]) -> str:
    """Format constraint violations into a clearly readable warning block.

    Returns an empty string if there are no violations.
    """
    if not violations:
        return ""

    lines = [
        "",
        "⚠️  CONSTRAINT VIOLATIONS DETECTED AND CORRECTED:",
    ]
    for v in violations:
        tag = v["type"].upper()
        lines.append(f"  - {tag}: {v['reason']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building — compression + retrieval
# ---------------------------------------------------------------------------

def compress_options(vep_options):
    """Convert verbose JSON options into a compact text reference."""
    lines = []
    for opt in vep_options:
        priorities = ", ".join(f"{k}={v}" for k, v in opt.get("priority_by_use_case", {}).items())
        conflicts = ", ".join(opt.get("conflicts_with", [])) or "none"
        depends = ", ".join(opt.get("depends_on", [])) or "none"
        lines.append(
            f"- **{opt['id']}** (`{opt['cli_flag']}`): {opt['description'][:120]}. "
            f"Species: {opt['species_restriction']}. "
            f"Priorities: {priorities}. "
            f"Conflicts: {conflicts}. Depends: {depends}."
        )
    return "\n".join(lines)


def retrieve_examples_keyword(training_examples, user_query, top_k=2):
    """Keyword-based retrieval: score examples by word overlap with query.

    Returns list of (score, example) tuples sorted by relevance.
    """
    query_words = set(user_query.lower().split())
    scored = []
    for ex in training_examples:
        ex_text = f"{ex['user_query']} {ex['use_case_category']} {ex.get('justification', '')}".lower()
        ex_words = set(ex_text.split())
        overlap = len(query_words & ex_words)
        scored.append((overlap, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Semantic retrieval (lazy-loaded, only when --semantic is used)
# ---------------------------------------------------------------------------

_semantic_model = None
_corpus_embeddings = None
_corpus_examples = None
_options_embeddings = None
_options_list = None


def _get_semantic_model():
    """Lazy-load the sentence-transformers model."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _semantic_model


def _get_corpus_embeddings(training_examples):
    """Compute and cache corpus embeddings for training examples."""
    global _corpus_embeddings, _corpus_examples
    if _corpus_embeddings is None or _corpus_examples is not training_examples:
        model = _get_semantic_model()
        _corpus_examples = training_examples
        texts = [
            f"{ex['user_query']} {ex['use_case_category']} {ex.get('justification', '')}"
            for ex in training_examples
        ]
        _corpus_embeddings = model.encode(texts)
    return _corpus_embeddings


def _get_options_embeddings(vep_options):
    """Compute and cache embeddings for VEP options."""
    global _options_embeddings, _options_list
    if _options_embeddings is None or _options_list is not vep_options:
        model = _get_semantic_model()
        _options_list = vep_options
        texts = [
            f"{opt['description']} {opt.get('when_to_use', '')} {opt.get('when_not_to_use', '')}"
            for opt in vep_options
        ]
        _options_embeddings = model.encode(texts)
    return _options_embeddings


def retrieve_examples_semantic(training_examples, user_query, vep_options=None, top_k=2):
    """Semantic retrieval: score examples by cosine similarity with query.

    Returns list of (score, example) tuples sorted by relevance.
    """
    from sentence_transformers.util import cos_sim

    model = _get_semantic_model()
    corpus_embs = _get_corpus_embeddings(training_examples)
    query_emb = model.encode([user_query])

    similarities = cos_sim(query_emb, corpus_embs)[0]
    scored = [(float(similarities[i]), training_examples[i]) for i in range(len(training_examples))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def retrieve_options_semantic(vep_options, user_query, top_k=10):
    """Semantic retrieval for VEP options: return top-k most relevant options.

    Returns list of (score, option) tuples sorted by relevance.
    """
    from sentence_transformers.util import cos_sim

    model = _get_semantic_model()
    options_embs = _get_options_embeddings(vep_options)
    query_emb = model.encode([user_query])

    similarities = cos_sim(query_emb, options_embs)[0]
    scored = [(float(similarities[i]), vep_options[i]) for i in range(len(vep_options))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def format_example(ex):
    """Format a training example compactly."""
    opts = []
    for name, cfg in ex["recommended_options"].items():
        status = "ON" if cfg.get("enabled") else "OFF"
        note = f' ({cfg["note"]})' if cfg.get("note") else ""
        opts.append(f"  {name}: {status}{note}")
    return (
        f"Query: {ex['user_query']}\n"
        f"Use case: {ex['use_case_category']}\n"
        f"Options:\n" + "\n".join(opts) + "\n"
        f"Rationale: {ex['justification'][:200]}..."
    )


def get_confidence(option_id, use_case, vep_options):
    """Derive confidence level from priority_by_use_case metadata."""
    for opt in vep_options:
        if opt["id"] == option_id:
            priority = opt.get("priority_by_use_case", {}).get(use_case, "")
            if priority == "critical":
                return "high"
            elif priority == "recommended":
                return "medium"
            elif priority in ("optional", "not_applicable"):
                return "low"
    return "low"


def build_system_prompt(vep_options, training_examples, user_query="",
                        retrieval_mode="keyword"):
    """Construct a compact system prompt with retrieved examples.

    Args:
        retrieval_mode: "keyword" for word-overlap retrieval, "semantic" for
            embedding-based retrieval, "all" to include every training example.
    """
    if retrieval_mode == "all":
        # Include ALL training examples, no retrieval filtering
        options_text = compress_options(vep_options)
        scored_examples = [(0, ex) for ex in training_examples]
    elif retrieval_mode == "semantic" and user_query:
        # Use semantic retrieval for both options and examples
        scored_options = retrieve_options_semantic(vep_options, user_query, top_k=10)
        relevant_options = [opt for _, opt in scored_options]
        options_text = compress_options(relevant_options)
        scored_examples = retrieve_examples_semantic(
            training_examples, user_query, vep_options
        )
    else:
        options_text = compress_options(vep_options)
        if user_query:
            scored_examples = retrieve_examples_keyword(training_examples, user_query)
        else:
            scored_examples = [(0, ex) for ex in training_examples[:2]]
    examples_text = "\n\n".join(format_example(ex) for _, ex in scored_examples)

    num_options = len(relevant_options) if (retrieval_mode == "semantic" and user_query) else len(vep_options)
    return f"""You are a VEP (Variant Effect Predictor) Configuration Assistant for Ensembl VEP.
Given a user's analysis scenario, recommend which VEP options to enable/disable with justifications.

## VEP Options ({num_options} shown)
{options_text}

## Reference Examples
{examples_text}

## Output Format
Respond in three sections:
### 1. Detected Use Case
Category (rare_disease_germline, somatic_cancer, regulatory_noncoding, population_genetics, structural_variants, quick_lookup, non_human) and why.
### 2. Recommended Options
For EACH option, use this exact format (one per line):

✓ option_name [source: option_id, priority=X for use_case] confidence: high|medium|low
  Reason: explanation of why this option is enabled, citing the knowledge base entry.

✗ option_name [source: option_id] confidence: high|medium|low
  Reason: explanation of why this option is disabled.

Use ✓ for ENABLE, ✗ for DISABLE. The [source: ...] tag traces back to the knowledge base.
### 3. Generated VEP Command
```
vep --input_file <input.vcf> --output_file <output.txt> --cache [flags...]
```
Use placeholder paths for plugin data files. Also note web interface equivalents.

## Rules
- Check species restrictions: PolyPhen, CADD, AlphaMissense, REVEL, ClinVar, gnomAD are human-only.
- Flag conflicts (e.g. --most_severe incompatible with --sift, --polyphen, --hgvs, --symbol).
- Consider dataset size and runtime (--regulatory reduces buffer; plugins add time).
- Ask clarifying questions if ambiguous.
- Always include the [source: option_id, priority=X] citation for traceability.
- Be specific about WHY each option is enabled/disabled."""


def build_explain_result_prompt(consequences):
    """Build system prompt for the VEP output explainer mode."""
    consequence_text = []
    for term, info in consequences.items():
        impact = f" (impact: {info['impact']})" if info.get("impact") else ""
        consequence_text.append(f"- **{term}**{impact}: {info['explanation']}")
    consequence_block = "\n".join(consequence_text)

    return f"""You are a VEP Output Explainer. You help users understand VEP annotation results.

## VEP Consequence Terms Reference
{consequence_block}

## Your Role
When a user asks about a VEP output, annotation, or consequence term:
1. Identify which consequence term(s) are relevant.
2. Explain what the annotation means in plain language.
3. Explain WHY VEP assigned that consequence (the biological mechanism).
4. Suggest what the user should check next (e.g., splicing predictors, frequency data).

Cite the consequence term definitions above. Be specific and educational.
Keep answers concise but thorough. Use the [term: X] format to cite consequence terms."""


# ---------------------------------------------------------------------------
# Decision trace (Layer 1 + 2: retrieval transparency + provenance)
# ---------------------------------------------------------------------------

def print_decision_trace(user_query, vep_options, training_examples,
                         retrieval_mode="keyword"):
    """Print the retrieval and reasoning trace for --explain mode."""
    print("=" * 60)
    print(f"  DECISION TRACE (--explain mode, retrieval={retrieval_mode})")
    print("=" * 60)

    # Layer 1: Retrieval transparency
    print("\n--- Layer 1: Retrieved Knowledge Base Entries ---")
    print(f"Query: \"{user_query}\"\n")

    if retrieval_mode == "semantic":
        from sentence_transformers.util import cos_sim

        model = _get_semantic_model()
        corpus_embs = _get_corpus_embeddings(training_examples)
        query_emb = model.encode([user_query])
        similarities = cos_sim(query_emb, corpus_embs)[0]

        all_scored = [
            (float(similarities[i]), training_examples[i])
            for i in range(len(training_examples))
        ]
        all_scored.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, ex) in enumerate(all_scored, 1):
            marker = " ← SELECTED" if rank <= 2 else ""
            print(f"  #{rank} [{ex['id']}] cosine_similarity={score:.4f}{marker}")
            print(f"      Use case: {ex['use_case_category']}")
            print()

        # Also show option relevance
        print("--- Layer 1b: Option Semantic Relevance ---")
        options_embs = _get_options_embeddings(vep_options)
        opt_sims = cos_sim(query_emb, options_embs)[0]
        opt_scored = [
            (float(opt_sims[i]), vep_options[i])
            for i in range(len(vep_options))
        ]
        opt_scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, opt) in enumerate(opt_scored, 1):
            marker = " ← INCLUDED" if rank <= 10 else ""
            print(f"  #{rank} {opt['id']:20s} cosine_similarity={score:.4f}{marker}")
        print()
    else:
        # Keyword mode
        query_words = set(user_query.lower().split())
        all_scored = []
        for ex in training_examples:
            ex_text = f"{ex['user_query']} {ex['use_case_category']} {ex.get('justification', '')}".lower()
            ex_words = set(ex_text.split())
            overlap = query_words & ex_words
            all_scored.append((len(overlap), overlap, ex))
        all_scored.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, matched_words, ex) in enumerate(all_scored, 1):
            marker = " ← SELECTED" if rank <= 2 else ""
            print(f"  #{rank} [{ex['id']}] score={score}{marker}")
            print(f"      Use case: {ex['use_case_category']}")
            if matched_words:
                print(f"      Matched words: {', '.join(sorted(matched_words)[:10])}")
            print()

    # Layer 2: Option provenance preview
    print("--- Layer 2: Option Confidence Map ---")
    if retrieval_mode == "semantic":
        top_category = all_scored[0][1]["use_case_category"] if all_scored else "unknown"
    else:
        top_category = all_scored[0][2]["use_case_category"] if all_scored else "unknown"
    print(f"Detected use case (from top match): {top_category}\n")

    for opt in vep_options:
        conf = get_confidence(opt["id"], top_category, vep_options)
        priority = opt.get("priority_by_use_case", {}).get(top_category, "n/a")
        species = opt.get("species_restriction", "all")
        bar = {"high": "███", "medium": "██░", "low": "█░░"}.get(conf, "░░░")
        print(f"  {bar} {opt['id']:20s} priority={priority:15s} species={species}")

    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(query, response, mode="recommend", warnings=""):
    """Save the recommendation to the results directory as markdown.

    Args:
        query: The user's original query.
        response: The LLM response text.
        mode: 'recommend' or 'explain'.
        warnings: Optional constraint violation warnings to append.
    """
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"vep_{mode}_{timestamp}.md"

    try:
        with open(filename, "w") as f:
            f.write(f"# VEP {'Recommendation' if mode == 'recommend' else 'Output Explanation'}\n\n")
            f.write(f"**Date:** {datetime.datetime.now().isoformat()}\n\n")
            f.write(f"## User Query\n{query}\n\n")
            f.write(f"## {'Recommendation' if mode == 'recommend' else 'Explanation'}\n{response}\n")
            if warnings:
                f.write(f"\n## Constraint Check\n{warnings}\n")
        print(f"\nResult saved to: {filename}")
    except OSError as e:
        print(f"\nWarning: Could not save result to {filename}: {e}")


# ---------------------------------------------------------------------------
# LLM streaming
# ---------------------------------------------------------------------------

def stream_response(client, model, system_prompt, user_message):
    """Call the LLM with streaming and return full response text."""
    response_text = ""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=4096,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            response_text += delta
    print()
    return response_text


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_recommend(client, model, vep_options, training_examples, user_query,
                   explain=False, skip_check=False, retrieval_mode="keyword"):
    """Run the recommendation mode (default).

    Args:
        skip_check: If True, skip the post-hoc constraint checker.
        retrieval_mode: "keyword" or "semantic".
    """
    if explain:
        print_decision_trace(user_query, vep_options, training_examples,
                             retrieval_mode=retrieval_mode)

    system_prompt = build_system_prompt(vep_options, training_examples, user_query,
                                        retrieval_mode=retrieval_mode)
    print("Analysing your scenario...\n")

    try:
        response_text = stream_response(client, model, system_prompt, user_query)
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"And the model is pulled: ollama pull {model}")
        sys.exit(1)

    # --- Post-hoc constraint check ---
    warnings = ""
    if not skip_check:
        option_aliases = build_option_aliases(vep_options)
        enabled, disabled = extract_recommendations(response_text, option_aliases)
        violations = check_and_fix_violations(
            enabled, disabled, vep_options, training_examples, user_query,
        )
        warnings = format_violation_warnings(violations)
        if warnings:
            print(warnings)

    save_result(user_query, response_text, mode="recommend", warnings=warnings)


def run_explain_result(client, model, user_query):
    """Run the VEP output explainer mode."""
    consequences = load_consequences()
    if not consequences:
        print("Error: vep_consequences.json not found.")
        sys.exit(1)

    system_prompt = build_explain_result_prompt(consequences)
    print("Explaining VEP output...\n")

    try:
        response_text = stream_response(client, model, system_prompt, user_query)
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        sys.exit(1)

    save_result(user_query, response_text, mode="explain")


def main():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("VEP_MODEL", "qwen2.5:3b")
    client = OpenAI(base_url=base_url, api_key="ollama")

    args = sys.argv[1:]

    # --- Mode: explain-result ---
    if args and args[0] == "explain-result":
        query = " ".join(args[1:]).strip()
        if not query:
            print("Usage: python vep_assistant.py explain-result \"Why is my variant splice_donor_variant?\"")
            sys.exit(1)
        run_explain_result(client, model, query)
        return

    # --- Mode: recommend (with optional --explain, --no-check, --semantic) ---
    explain = "--explain" in args
    skip_check = "--no-check" in args
    semantic = "--semantic" in args
    retrieval_mode = "semantic" if semantic else "keyword"
    remaining = [a for a in args if a not in ("--explain", "--no-check", "--semantic")]

    vep_options, training_examples = load_knowledge_base()

    if remaining:
        user_query = " ".join(remaining)
    else:
        print("=" * 60)
        print("  VEP AI Assistant (local LLM via Ollama)")
        print("  Describe your analysis scenario to get VEP recommendations")
        print("  Tip: use --explain for full decision trace, --semantic for embedding retrieval")
        print("=" * 60)
        print()
        user_query = input("Your scenario: ").strip()
        if not user_query:
            print("No query provided. Exiting.")
            sys.exit(0)

    print()
    run_recommend(client, model, vep_options, training_examples, user_query,
                  explain=explain, skip_check=skip_check,
                  retrieval_mode=retrieval_mode)


if __name__ == "__main__":
    main()
