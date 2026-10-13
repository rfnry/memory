import rfnry_memory

EXPECTED = {
    "Memory",
    "MemoryTools",
    "Tool",
    "Tools",
    "PostgresStorage",
    "Kind",
    "Ref",
    "Source",
    "Section",
    "Chunk",
    "Story",
    "Entity",
    "Relation",
    "EntityIn",
    "RelationIn",
    "Hit",
    "Hits",
    "Judgement",
    "JudgementName",
    "PillarReport",
    "Ingested",
    "Remembered",
    "Synced",
    "Language",
    "SourceIn",
    "Counts",
    "CorpusDescription",
    "CorpusFingerprint",
    "Chunking",
    "VectorIndex",
    "Neighborhood",
    "DocumentEmbeddingModel",
    "EmbeddingModel",
    "GenerationModel",
    "DecisionModel",
    "ReplyLike",
    "DecidedLike",
    "AnswerLike",
    "QuestionLike",
    "Message",
    "ToolSpec",
    "ToolCall",
    "Reply",
    "Usage",
    "State",
    "Text",
    "Image",
    "Audio",
    "Document",
    "Part",
    "Content",
    "Choice",
    "Score",
    "Check",
    "Question",
    "ChoiceAnswer",
    "ScoreAnswer",
    "CheckAnswer",
    "Answer",
    "Decided",
    "confidence",
    "MemoryException",
    "ConfigurationError",
    "InputError",
    "ModelError",
    "StorageError",
    "NotFoundError",
}


def test_public_names_are_exactly_these() -> None:
    assert set(rfnry_memory.__all__) == EXPECTED
    for name in EXPECTED:
        assert getattr(rfnry_memory, name) is not None


async def test_deleting_and_verifying_count_rows_the_same_way(
    storage: rfnry_memory.PostgresStorage,
) -> None:
    memory = rfnry_memory.Memory(storage)
    source = (await memory.ingest("acme", name="a", text="alpha")).source
    deleted = await memory.delete_source("acme", source.source_id)
    residue = await memory.verify_absent("acme")
    assert type(deleted) is type(residue) is rfnry_memory.Counts
    assert (deleted.sources, deleted.chunks, deleted.clean) == (1, 1, False) and residue.clean
