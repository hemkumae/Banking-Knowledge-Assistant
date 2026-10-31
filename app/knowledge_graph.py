import networkx as nx


class BankingKnowledgeGraph:
    """Small explicit graph model: entities are nodes; relationships are edges."""

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

    def neighbors(self, entity: str):
        if entity not in self.graph:
            return {"entity": entity, "found": False, "relationships": []}
        relationships = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            relationships.append({"relation": data["relation"], "target": target})
        for source, _, data in self.graph.in_edges(entity, data=True):
            relationships.append({"relation": data["relation"], "source": source})
        return {"entity": entity, "found": True, "relationships": relationships}

    def shortest_path(self, source: str, target: str):
        if source not in self.graph or target not in self.graph:
            return {"found": False, "path": []}
        try:
            path = nx.shortest_path(self.graph.to_undirected(), source, target)
            return {"found": True, "path": path}
        except nx.NetworkXNoPath:
            return {"found": False, "path": []}
