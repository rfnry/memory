# rfnry-memory

rfnry-memory is a hybrid-retrieval memory system for AI agents.

It holds both halves of what an agent knows: the knowledge it is given — documents, the job RAG does — and the memory it keeps — facts and events that change over time. One search runs keyword, semantic and graph search together and returns one ranked list, and every memory knows when it was true and when it was learned.

The contract is [`docs/memory.md`](../../docs/memory.md).
The reference is [`docs/api.md`](../../docs/api.md).
