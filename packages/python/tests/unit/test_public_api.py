import rfnry_memory

EXPECTED = {
    "Memory", "MemoryTools", "Tool", "Tools", "Storage", "InMemoryStorage", "PostgresStorage",
    "Source", "Section", "Chunk", "Story", "Entity", "Relation", "EntityIn", "RelationIn",
    "Hit", "Hits", "Judgement", "PillarReport", "Ingested", "Deleted", "Residue", "Neighborhood", "Window",
    "EmbeddingModel", "GenerationModel", "DecisionModel",
    "Message", "ToolSpec", "ToolCall", "Reply", "Usage", "State",
    "Choice", "Score", "Check", "Question", "ChoiceAnswer", "ScoreAnswer", "CheckAnswer", "Answer", "Decided",
    "confidence", "fuse_by_rank", "chunk_text",
    "MemoryException", "ConfigurationError", "InputError", "ModelError", "StorageError", "NotFoundError",
}


def test_public_names_are_exactly_these() -> None:
    assert set(rfnry_memory.__all__) == EXPECTED
    for name in EXPECTED:
        assert getattr(rfnry_memory, name) is not None
