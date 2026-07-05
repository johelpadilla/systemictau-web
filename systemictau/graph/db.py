import os
try:
    from neo4j import GraphDatabase
except ImportError:
    pass

class KnowledgeGraphService:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        # Only initialize if neo4j is available
        if 'GraphDatabase' in globals():
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        else:
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def persist_ontological_ascent(self, tenant_id: str, t_star: int, tau_value: float, description: str):
        """
        Creates a (System)-[:UNDERWENT]->(Ascent) relationship in the Neo4j Knowledge Graph.
        Allows for traversing causality chains of ontological transitions.
        """
        if not self.driver:
            print("Neo4j driver not initialized. Skipping graph persistence.")
            return
            
        query = """
        MERGE (s:System {tenant_id: $tenant_id})
        CREATE (a:OntologicalAscent {
            t_star: $t_star, 
            tau: $tau_value, 
            description: $description,
            timestamp: timestamp()
        })
        CREATE (s)-[:UNDERWENT]->(a)
        RETURN id(a) as node_id
        """
        with self.driver.session() as session:
            result = session.run(query, tenant_id=tenant_id, t_star=t_star, tau_value=tau_value, description=description)
            return result.single()["node_id"]
            
    def get_historical_context(self, tenant_id: str, limit: int = 5) -> list:
        """
        Retrieves the last N transitions for a specific tenant to provide Graph-RAG context.
        """
        if not self.driver:
            return []
            
        query = """
        MATCH (s:System {tenant_id: $tenant_id})-[:UNDERWENT]->(a:OntologicalAscent)
        RETURN a.t_star AS t_star, a.tau AS tau, a.description AS description, a.timestamp AS timestamp
        ORDER BY a.timestamp DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, tenant_id=tenant_id, limit=limit)
            return [record.data() for record in result]
            
    def get_epistemic_history(self, tenant_id: str, limit: int = 5) -> list:
        if not self.driver:
            return []
            
        query = """
        MATCH (s:System {tenant_id: $tenant_id})-[:UNDERWENT]->(a:OntologicalAscent)
        OPTIONAL MATCH (a)-[:CAUSED_BY_HYPOTHESIS]->(h:HypothesisNode)
        OPTIONAL MATCH (e:EvidenceNode)-[:APPLIES_TO]->(h)
        RETURN a.t_star AS t_star, a.tau AS tau, a.description AS description, 
               h.claim AS claim, h.confidence AS confidence, 
               e.source AS source, e.data_summary AS summary, e.supports_hypothesis AS supports
        ORDER BY a.timestamp DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, tenant_id=tenant_id, limit=limit)
            return [record.data() for record in result]
            
    def persist_agent_report(self, ascent_node_id: int, report_text: str):
        """
        Attaches an LLM generated ReportNode to an AscentNode in the Graph.
        """
        if not self.driver:
            return
            
        query = """
        MATCH (a:OntologicalAscent) WHERE id(a) = $node_id
        CREATE (r:ReportNode {
            content: $content,
            timestamp: timestamp()
        })
        CREATE (a)-[:SYNTHESIZED_INTO]->(r)
        """
        with self.driver.session() as session:
            session.run(query, node_id=ascent_node_id, content=report_text)
            
    def persist_hypothesis(self, ascent_node_id: int, claim: str, confidence: float) -> int:
        if not self.driver:
            return -1
        query = """
        MATCH (a:OntologicalAscent) WHERE id(a) = $node_id
        CREATE (h:HypothesisNode {
            claim: $claim,
            confidence: $confidence,
            timestamp: timestamp()
        })
        CREATE (a)-[:CAUSED_BY_HYPOTHESIS]->(h)
        RETURN id(h) AS h_id
        """
        with self.driver.session() as session:
            result = session.run(query, node_id=ascent_node_id, claim=claim, confidence=confidence)
            record = result.single()
            return record["h_id"] if record else -1

    def persist_evidence(self, hypothesis_node_id: int, source: str, summary: str, supports: bool):
        if not self.driver:
            return
        query = """
        MATCH (h:HypothesisNode) WHERE id(h) = $node_id
        CREATE (e:EvidenceNode {
            source: $source,
            data_summary: $summary,
            supports_hypothesis: $supports
        })
        CREATE (e)-[:APPLIES_TO]->(h)
        """
        with self.driver.session() as session:
            session.run(query, node_id=hypothesis_node_id, source=source, summary=summary, supports=supports)

    def persist_tool_usage(self, evidence_node_id: int, tool_name: str, version: str = "1.0"):
        if not self.driver:
            return
        query = """
        MATCH (e:EvidenceNode) WHERE id(e) = $node_id
        MERGE (t:ToolNode {name: $tool_name, version: $version})
        CREATE (e)-[:COLLECTED_VIA]->(t)
        """
        with self.driver.session() as session:
            session.run(query, node_id=evidence_node_id, tool_name=tool_name, version=version)
            
    def correlate_hypotheses(self, h1_id: int, h2_id: int, relationship_type: str):
        """
        relationship_type should be 'CORROBORATES' or 'CONTRADICTS'
        """
        if not self.driver or relationship_type not in ["CORROBORATES", "CONTRADICTS"]:
            return
        query = f"""
        MATCH (h1:HypothesisNode), (h2:HypothesisNode)
        WHERE id(h1) = $h1_id AND id(h2) = $h2_id
        CREATE (h1)-[:{relationship_type}]->(h2)
        """
        with self.driver.session() as session:
            session.run(query, h1_id=h1_id, h2_id=h2_id)

    def persist_universal_theory(self, macro_claim: str, isomorphisms: int, confidence: float):
        if not self.driver:
            return None
        query = """
        CREATE (ut:UniversalTheoryNode {
            macro_claim: $macro_claim,
            isomorphisms_found: $isomorphisms,
            confidence: $confidence,
            timestamp: timestamp()
        })
        RETURN id(ut) as node_id
        """
        with self.driver.session() as session:
            result = session.run(query, macro_claim=macro_claim, isomorphisms=isomorphisms, confidence=confidence)
            record = result.single()
            return record["node_id"] if record else None
            
    def get_epistemic_voids(self):
        """Find hypotheses that have low confidence or lack corroborating evidence (v7.0 Proactive Loop)."""
        if not self.driver:
            return []
        query = """
        MATCH (h:HypothesisNode)
        WHERE h.confidence < 0.7 OR NOT (h)-[:APPLIES_TO]->(:EvidenceNode)
        RETURN id(h) as h_id, h.claim as claim, h.confidence as confidence
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [{"id": r["h_id"], "claim": r["claim"], "confidence": r["confidence"]} for r in result]
