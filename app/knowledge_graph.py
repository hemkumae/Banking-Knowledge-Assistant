import networkx as nx


class BankingKnowledgeGraph:
    """Explicit directed graph: entities are nodes and typed relations are edges."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._seed_graph()

    def _seed_graph(self):
        nodes = [
            ("Savings Account", "Product"), ("Customer", "Entity"),
            ("Interest", "Concept"), ("Fixed Deposit", "Product"),
            ("Personal Loan", "Product"), ("Credit History", "Concept"),
            ("KYC", "Process"), ("Identity Document", "Document"),
            ("NEFT", "PaymentMethod"), ("IMPS", "PaymentMethod"),
            ("Bank Account", "Entity"), ("Credit Card", "Product"),
            ("Credit Limit", "Concept"), ("Repayment", "Process")
        ]
        for name, kind in nodes:
            self.graph.add_node(name, entity_type=kind)

        edges = [
            ("Customer", "Savings Account", "OPENS"),
            ("Savings Account", "Interest", "MAY_EARN"),
            ("Customer", "Fixed Deposit", "INVESTS_IN"),
            ("Fixed Deposit", "Interest", "EARNS"),
            ("Customer", "Personal Loan", "APPLIES_FOR"),
            ("Personal Loan", "Credit History", "ASSESSES"),
            ("Customer", "KYC", "COMPLETES"),
            ("KYC", "Identity Document", "MAY_REQUIRE"),
            ("NEFT", "Bank Account", "TRANSFERS_TO"),
            ("IMPS", "Bank Account", "TRANSFERS_TO"),
            ("Customer", "Credit Card", "USES"),
            ("Credit Card", "Credit Limit", "HAS"),
            ("Customer", "Repayment", "MAKES")
        ]
        for source, target, relation in edges:
            self.graph.add_edge(source, target, relation=relation)

    def entities(self, entity_type=None):
        """List entities, optionally filtered by their type."""
        return [
            {"name": name, "entity_type": data.get("entity_type", "Unknown")}
            for name, data in self.graph.nodes(data=True)
            if entity_type is None or data.get("entity_type", "").lower() == entity_type.lower()
        ]

    def neighbors(self, entity: str, direction="both"):
        if entity not in self.graph:
            return {"entity": entity, "found": False, "relationships": []}
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")

        relationships = []
        if direction in {"outgoing", "both"}:
            for _, target, data in self.graph.out_edges(entity, data=True):
                relationships.append({"direction": "outgoing", "source": entity,
                                       "relation": data["relation"], "target": target})
        if direction in {"incoming", "both"}:
            for source, _, data in self.graph.in_edges(entity, data=True):
                relationships.append({"direction": "incoming", "source": source,
                                       "relation": data["relation"], "target": entity})
        return {"entity": entity, "found": True, "relationships": relationships}

    def relationships(self, relation=None):
        """Return typed edges, optionally filtered by relation name."""
        rows = []
        for source, target, data in self.graph.edges(data=True):
            if relation is None or data["relation"].lower() == relation.lower():
                rows.append({"source": source, "relation": data["relation"], "target": target})
        return rows

    def shortest_path(self, source: str, target: str, directed=True):
        if source not in self.graph or target not in self.graph:
            return {"found": False, "path": []}
        graph = self.graph if directed else self.graph.to_undirected()
        try:
            path = nx.shortest_path(graph, source, target)
            return {"found": True, "path": path, "directed": directed}
        except nx.NetworkXNoPath:
            return {"found": False, "path": [], "directed": directed}
