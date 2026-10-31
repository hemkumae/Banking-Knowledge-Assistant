from app.knowledge_graph import BankingKnowledgeGraph

def test_entities_filtered_by_type():
    graph = BankingKnowledgeGraph()
    products = graph.entities("product")
    assert products and all(item["entity_type"] == "Product" for item in products)

def test_outgoing_relationship_has_source_relation_target():
    graph = BankingKnowledgeGraph()
    result = graph.neighbors("Customer", direction="outgoing")
    assert {"direction": "outgoing", "source": "Customer", "relation": "OPENS", "target": "Savings Account"} in result["relationships"]

def test_incoming_relationship():
    graph = BankingKnowledgeGraph()
    result = graph.neighbors("Savings Account", direction="incoming")
    assert result["relationships"][0]["source"] == "Customer"

def test_directed_path_and_unknown_entity():
    graph = BankingKnowledgeGraph()
    assert graph.shortest_path("Customer", "Interest")["found"]
    assert not graph.shortest_path("Interest", "Customer")["found"]
    assert not graph.shortest_path("Missing", "Customer")["found"]

def test_invalid_direction():
    graph = BankingKnowledgeGraph()
    try:
        graph.neighbors("Customer", direction="sideways")
        assert False, "Expected ValueError"
    except ValueError:
        pass
