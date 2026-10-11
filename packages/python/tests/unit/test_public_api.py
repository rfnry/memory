import re

import rfnry_memory
from rfnry_memory.storage.storage import Storage

EXPECTED = {
    "Memory",
    "MemoryTools",
    "Tool",
    "Tools",
    "Storage",
    "InMemoryStorage",
    "PostgresStorage",
    "SourceWrite",
    "GraphWrite",
    "Window",
    "Kind",
    "Ref",
    "Leg",
    "Scope",
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
    "CorpusFingerprint",
    "Chunking",
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


def test_a_storage_of_your_own_is_typed_from_exported_names_alone() -> None:
    builtin = {"Any", "None", "str", "int", "float", "bool", "list", "dict", "tuple", "datetime"}
    named = {
        identifier
        for name, method in vars(Storage).items()
        if callable(method) and not name.startswith("_")
        for annotation in method.__annotations__.values()
        for identifier in re.findall(r"[A-Za-z_]\w*", str(annotation))
    }
    assert named - builtin <= set(rfnry_memory.__all__)


def test_public_names_are_exactly_these() -> None:
    assert set(rfnry_memory.__all__) == EXPECTED
    for name in EXPECTED:
        assert getattr(rfnry_memory, name) is not None


async def test_deleting_and_verifying_count_rows_the_same_way() -> None:
    memory = rfnry_memory.Memory(rfnry_memory.InMemoryStorage())
    source = (await memory.ingest("acme", name="a", text="alpha")).source
    deleted = await memory.delete_source("acme", source.source_id)
    residue = await memory.verify_absent("acme")
    assert type(deleted) is type(residue) is rfnry_memory.Counts
    assert (deleted.sources, deleted.chunks, deleted.clean) == (1, 1, False) and residue.clean
