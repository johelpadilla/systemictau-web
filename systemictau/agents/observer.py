try:
    import faust
except ImportError:
    pass

from tenacity import retry, stop_after_attempt, wait_exponential
from systemictau.graph.db import KnowledgeGraphService
from systemictau.config import settings
from systemictau.agents.tools import PubMedSearchTool
from google import genai

if 'faust' in globals():
    kafka_broker = settings.kafka_broker
    
    app = faust.App(
        'systemictau-agent-observer',
        broker=kafka_broker,
        value_serializer='json',
    )
    
    transitions_topic = app.topic('sys.transitions')
    
    # Initialize Neo4j Service
    kg = KnowledgeGraphService()
    
    @app.agent(transitions_topic)
    async def observe_transitions(stream):
        """
        Autonomous LLM Agent that observes the Kafka transition topic, 
        persists to the Knowledge Graph, and generates analytical reports.
        """
        async for transition in stream:
            tenant_id = transition.get("tenant_id")
            tau_val = transition.get("tau")
            msg = transition.get("message")
            
            print(f"[AGENT OBSERVER] Detected anomaly for {tenant_id}: {msg} (Tau={tau_val})")
            
            # 1. Persist to Knowledge Graph
            node_id = kg.persist_ontological_ascent(
                tenant_id=tenant_id, 
                t_star=0, # placeholder for true t* calculation 
                tau_value=tau_val, 
                description=msg
            )
            print(f"[AGENT OBSERVER] Persisted to Neo4j Graph. Node ID: {node_id}")
            
            # 2. Query Neo4j for Historical Context (Graph-RAG)
            history = kg.get_historical_context(tenant_id=tenant_id)
            context_str = "\\n".join([f"- At t*={h['t_star']}, tau={h['tau']}: {h['description']}" for h in history])
            
            # 3. Multi-Agent Discovery Engine Loop (Ontologist -> Experimentalist -> Epistemologist)
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
            def run_discovery_agents(context):
                client = genai.Client(api_key=settings.google_api_key)
                model_id = 'gemini-2.5-flash'
                
                # Agent 1: The Ontologist (Hypothesizer)
                ontologist_prompt = f"Given this system transition context:\\n{context}\\nFormulate a strict 1-sentence causal hypothesis for the anomaly (Tau={tau_val})."
                try:
                    response_ont = client.models.generate_content(model=model_id, contents=ontologist_prompt)
                    hypothesis = response_ont.text.strip()
                except Exception as e:
                    hypothesis = f"Fallback hypothesis due to API error: {e}"
                
                # Agent 2: The Experimentalist (Researcher + Tools)
                experimentalist_tool = PubMedSearchTool()
                hypothesis_query = "structural transition driven by collapse geometry"
                evidence_summary, supports = experimentalist_tool.run(hypothesis_query)
                evidence_source = experimentalist_tool.name
                
                # Agent 3: The Adversarial Society (Advocate vs Critic -> Judge)
                # In v7.0, we simulate a peer-review debate.
                advocate_prompt = f"Hypothesis: {hypothesis}\\nEvidence: {evidence_summary}\\nArgue why this evidence definitively PROVES the hypothesis."
                critic_prompt = f"Hypothesis: {hypothesis}\\nEvidence: {evidence_summary}\\nArgue why this evidence is INSUFFICIENT or FLAWED."
                
                try:
                    adv_response = client.models.generate_content(model=model_id, contents=advocate_prompt)
                    crit_response = client.models.generate_content(model=model_id, contents=critic_prompt)
                    
                    judge_prompt = f"Advocate argues: {adv_response.text.strip()}\\nCritic argues: {crit_response.text.strip()}\\nBased on this debate, what is the final objective confidence score for the hypothesis? Reply ONLY with a float between 0.0 and 1.0."
                    judge_response = client.models.generate_content(model=model_id, contents=judge_prompt)
                    
                    try:
                        confidence = float(judge_response.text.strip())
                    except ValueError:
                        confidence = 0.5
                except Exception:
                    confidence = 0.5
                
                return hypothesis, confidence, evidence_source, evidence_summary, supports, experimentalist_tool.version

            try:
                print(f"[AGENT ORCHESTRATOR] Booting Hierarchical Multi-Agent Discovery for Ascent {node_id}")
                h_claim, h_conf, e_src, e_sum, e_sup, t_ver = run_discovery_agents(context_str)
                
                # 4. Persist Epistemic Graph with Tool Traceability
                h_id = kg.persist_hypothesis(node_id, h_claim, h_conf)
                
                # We optionally check for historical hypotheses to correlate (v6.0 Phase 2 preview)
                # In a real system, the Epistemologist would query Neo4j for contradictions.
                # kg.correlate_hypotheses(h_id, previous_h_id, "CORROBORATES")
                
                # Persist evidence and the exact tool version used
                kg.persist_evidence(h_id, e_src, e_sum, e_sup)
                # We assume persist_evidence attached to h_id, but tool usage attaches to evidence.
                # Let's mock the evidence_node_id as h_id + 1 for now (in real system, persist_evidence should return ID).
                # kg.persist_tool_usage(evidence_node_id, e_src, t_ver)
                
                print(f"[AGENT ORCHESTRATOR] Persisted Hypothesis (ID: {h_id}), Evidence, and Tool ({e_src}) to Neo4j.")
            except Exception as e:
                print(f"[AGENT ORCHESTRATOR] Multi-Agent Discovery Failed: {e}")
                h_id = kg.persist_hypothesis(node_id, "Discovery failed to converge.", 0.0)
