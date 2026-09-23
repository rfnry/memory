import voyageai
from anthropic import AsyncAnthropic
from rfnry_memory import Reply, ToolCall, Usage


class Voyage:
    model, dimension = "voyage-4-lite", 1024

    def __init__(self):
        self.client = voyageai.AsyncClient(max_retries=3)

    async def embed(self, texts):
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings


class Claude:
    def __init__(self, model="claude-sonnet-5", max_tokens=4096, effort="low"):
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self.client = AsyncAnthropic()

    async def generate(self, system, messages, tools):
        response = await self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=[wire(m) for m in messages],
            tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools],
            output_config={"effort": self.effort})
        blocks = [block.model_dump(exclude_none=True) for block in response.content]
        usage = response.usage
        return Reply(
            text="".join(block["text"] for block in blocks if block["type"] == "text"),
            tool_calls=[ToolCall(id=b["id"], name=b["name"], args=b["input"]) for b in blocks if b["type"] == "tool_use"],
            usage=Usage(input=usage.input_tokens, output=usage.output_tokens),
            stop_reason="tool_use" if response.stop_reason == "tool_use" else "end_turn",
            model=response.model,
        )


def wire(message):
    if message.role == "tool":
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.tool_call_id,
                                             "content": message.content, "is_error": message.is_error}]}
    return {"role": message.role, "content": message.content}
