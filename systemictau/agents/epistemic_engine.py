from google import genai
from systemictau.config import settings

TAU_KNOWLEDGE_BASE = """
SYSTEMIC TAU (τ_s) THEORETICAL FRAMEWORK:
1. Ontological Axiom: Complex systems are not just data points; they are bounded by 'extramental coherence' (the physical/structural reality of their connections).
2. The Tau Metric: τ_s quantifies this coherence over time. When τ_s hits a critical mass threshold (M*), a phase transition (t*) is mathematically inevitable.
3. Topological Reorganization: At t*, the system cannot sustain its current complexity and undergoes 'cascading failure geometry' or 'structural bifurcation' to survive, shedding excess entropy.
4. Entropic Decay: The buildup of systemic chaos (S_e) acts as the catalyst for these reorganizations.
You MUST interpret all data strictly through these axioms. Do not use generic explanations. Use formal terminology: extramental coherence, topological reorganization, structural breakpoint, entropic decay.
"""

def run_discovery_engine_sync(context: str, tau_val: float, update_callback=None):
    """
    The True Systemic Tau Epistemic Engine (Standalone Version).
    Orchestrates the Ontologist, Advocate, Critic, and Judge agents.
    
    Args:
        context (str): The domain and context string (e.g. data column anomalies).
        tau_val (float): The mathematical structural break magnitude.
        update_callback (callable): Optional callback function that takes a string 
                                    to stream updates to the GUI.
    """
    
    def log(msg):
        if update_callback:
            update_callback(msg)
        else:
            print(msg)

    # Validate API Key
    api_key = settings.google_api_key
    if not api_key or api_key == "DUMMY_GEMINI_KEY":
        raise ValueError("API key not valid. Please pass a valid API key.")
        
    client = genai.Client(api_key=api_key)
    model_id = 'gemini-2.5-flash'
    
    # ---------------------------------------------------------
    # AGENT 1: The Ontologist (Hypothesizer)
    # ---------------------------------------------------------
    log("\\n[Agent 1: Ontologist] Formulating initial hypothesis based on Tau transition...\\n")
    ontologist_prompt = (
        f"{TAU_KNOWLEDGE_BASE}\\n\\n"
        f"Context: {context}\\n"
        f"Mathematical Transition (Tau_s): {tau_val}\\n\\n"
        "You are the Systemic Tau AI Ontologist. According to the Systemic Tau paradigm, Tau (τ_s) quantifies "
        "the discrete extramental coherence of a complex system across time. A critical phase transition (t*) "
        "occurs when this coherence crosses a structural mass threshold, forcing the system to shed complexity "
        "or undergo topological reorganization to survive.\\n\\n"
        "Formulate a strict, 1-sentence causal scientific hypothesis explaining why this specific variable "
        "experienced a topological reorganization (t*). Do not hallucinate generic stories; focus strictly on "
        "systemic coherence, entropic decay, and structural breakpoints."
    )
    try:
        response_ont = client.models.generate_content(model=model_id, contents=ontologist_prompt)
        hypothesis = response_ont.text.strip()
        log(f"      -> Hypothesis: {hypothesis}\\n")
    except Exception as e:
        raise RuntimeError(f"Ontologist Agent Error: {e}")

    # ---------------------------------------------------------
    # AGENT 2: The Experimentalist (Mocked Data Retrieval)
    # ---------------------------------------------------------
    log("\\n[Agent 2: Experimentalist] Fetching empirical evidence...\\n")
    # Simulating the PubMedSearchTool output for the desktop app so it doesn't hang on network scraping
    evidence_summary = (
        "Empirical retrieval [Simulated PubMed/arXiv]: Time-series datasets across multiple "
        "complex domains confirm that a rapid deviation in systemic coherence (high τ_s) immediately "
        "precedes macroscopic topological collapse and irreversible phase bifurcations."
    )
    log(f"      -> Evidence: {evidence_summary}\\n")

    # ---------------------------------------------------------
    # AGENT 3A: The Advocate (Defense)
    # ---------------------------------------------------------
    log("\\n[Agent 3A: Advocate] Defending the hypothesis...\\n")
    advocate_prompt = (
        f"{TAU_KNOWLEDGE_BASE}\\n\\n"
        f"Hypothesis: {hypothesis}\\n"
        f"Evidence: {evidence_summary}\\n\\n"
        "You are the Systemic Tau Advocate. Argue aggressively in 2-3 sentences why this empirical evidence "
        "definitively proves the hypothesis. Use the formal language of topological reorganization, "
        "systemic coherence, and causal bifurcation points."
    )
    try:
        adv_response = client.models.generate_content(model=model_id, contents=advocate_prompt)
        advocate_arg = adv_response.text.strip()
        log(f"      -> Defense: {advocate_arg}\\n")
    except Exception as e:
        raise RuntimeError(f"Advocate Agent Error: {e}")

    # ---------------------------------------------------------
    # AGENT 3B: The Critic (Attack)
    # ---------------------------------------------------------
    log("\\n[Agent 3B: Critic] Attacking the hypothesis...\\n")
    critic_prompt = (
        f"{TAU_KNOWLEDGE_BASE}\\n\\n"
        f"Systemic Tau Hypothesis: {hypothesis}\\n"
        f"Evidence: {evidence_summary}\\n\\n"
        "You are the Systemic Tau Critic. Argue aggressively in 2-3 sentences why this evidence "
        "is mathematically INSUFFICIENT to prove the hypothesis. Attack the lack of rigorous correlation "
        "between the observed entropic decay and the claimed topological reorganization."
    )
    try:
        crit_response = client.models.generate_content(model=model_id, contents=critic_prompt)
        critic_arg = crit_response.text.strip()
        log(f"      -> Attack: {critic_arg}\\n")
    except Exception as e:
        raise RuntimeError(f"Critic Agent Error: {e}")

    # ---------------------------------------------------------
    # AGENT 4: The Judge (Verdict)
    # ---------------------------------------------------------
    log("\\n[Agent 4: Judge] Weighing arguments and issuing verdict...\\n")
    judge_prompt = (
        f"Advocate argues: {advocate_arg}\\n"
        f"Critic argues: {critic_arg}\\n\\n"
        "Based on this debate, what is the final objective confidence score for the hypothesis? "
        "Reply ONLY with a float between 0.0 and 1.0. No other text."
    )
    try:
        judge_response = client.models.generate_content(model=model_id, contents=judge_prompt)
        try:
            confidence = float(judge_response.text.strip())
        except ValueError:
            confidence = 0.50
        log(f"      -> Final Confidence Score (p*): {confidence}\\n")
    except Exception as e:
        confidence = 0.50
        log(f"      -> Judge Agent Error: {e}. Defaulting to {confidence}\\n")

    return hypothesis, confidence
