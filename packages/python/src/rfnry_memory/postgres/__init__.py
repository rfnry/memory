from rfnry_memory.postgres.corpus import Corpus
from rfnry_memory.postgres.graph import Graph
from rfnry_memory.postgres.schema import Schema
from rfnry_memory.postgres.search import Search
from rfnry_memory.postgres.sources import Sources
from rfnry_memory.postgres.stories import Stories
from rfnry_memory.postgres.vectors import Vectors


class PostgresStorage(Schema, Sources, Stories, Graph, Search, Corpus, Vectors):
    pass
