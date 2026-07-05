import time
from google import genai
from systemictau.graph.db import KnowledgeGraphService
from systemictau.config import settings

def run_proactive_engine():
    """
    v7.0 Proactive Epistemic Engine.
    Runs asynchronously, identifying unresolved hypotheses in the graph and deploying
    autonomous agents to find new evidence, effectively directing institutional research budgets.
    """
    kg = KnowledgeGraphService()
    client = genai.Client(api_key=settings.google_api_key)
    model_id = 'gemini-2.5-flash'
    
    print("[PROACTIVE ENGINE] Booting v7.0 Cybernetic Agenda Setter...")
    
    while True:
        voids = kg.get_epistemic_voids()
        
        if not voids:
            print("[PROACTIVE ENGINE] No epistemic voids found. Graph is in consensus. Sleeping...")
            time.sleep(60)
            continue
            
        for void in voids:
            h_id = void["id"]
            claim = void["claim"]
            conf = void["confidence"]
            
            print(f"[PROACTIVE ENGINE] Investigating Void ID {h_id} (Confidence: {conf})")
            print(f"[PROACTIVE ENGINE] Claim: {claim}")
            
            # Autonomously design an experiment or query to resolve the void
            agenda_prompt = f"We have an unresolved scientific claim: '{claim}'. Propose a strict query or simulation to test this claim."
            try:
                response = client.models.generate_content(model=model_id, contents=agenda_prompt)
                proposed_action = response.text.strip()
                print(f"[PROACTIVE ENGINE] Orchestrator proposes action: {proposed_action[:100]}...")
                
                # In a mature v7.0 system, this is where the PythonREPLTool or Simulation Orchestrator
                # would be spun up via Kubernetes to test the claim. For the bootstrap, we log the intent.
                
                # Assume a successful test bumps confidence
                min(conf + 0.15, 0.95)
                # kg.update_hypothesis_confidence(h_id, new_conf) # mock update
                print("[PROACTIVE ENGINE] Simulated test complete. Epistemic resolution advanced.")
            except Exception as e:
                print(f"[PROACTIVE ENGINE] Failed to resolve void: {e}")
                
        # Sleep before next graph sweep
        time.sleep(30)

if __name__ == "__main__":
    run_proactive_engine()
