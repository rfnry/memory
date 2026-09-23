import os

import voyageai
from anthropic import AsyncAnthropic
from rfnry_harness import Reply, ToolCall, Usage


class VoyageProvider:
    dimension = 1024

    def __init__(self, model="voyage-4-lite"):
        self.model = model
        self.client = voyageai.AsyncClient(api_key=os.environ["VOYAGE_API_KEY"], max_retries=3)

    async def embed(self, texts):
        response = await self.client.embed(texts, model=self.model)
        return response.embeddings


class AnthropicProvider:
    def __init__(self, model="claude-sonnet-5", max_tokens=4096, effort="low"):
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def generate(self, system, messages, tools):
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": wire(messages),
            "output_config": {"effort": self.effort},
        }
        if tools:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        response = await self.client.messages.create(**request)
        blocks = [block.model_dump(exclude_none=True) for block in response.content]
        thinking = any(block["type"] in ("thinking", "redacted_thinking") for block in blocks)
        usage = response.usage
        return Reply(
            text="".join(block["text"] for block in blocks if block["type"] == "text"),
            tool_calls=[
                ToolCall(id=b["id"], name=b["name"], args=b["input"])
                for b in blocks
                if b["type"] == "tool_use"
            ],
            usage=Usage(
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_creation=usage.cache_creation_input_tokens or 0,
                cache_read=usage.cache_read_input_tokens or 0,
            ),
            stop_reason=response.stop_reason
            if response.stop_reason in ("tool_use", "max_tokens")
            else "end_turn",
            model=response.model,
            reasoning=blocks if thinking else None,
        )


def wire(messages):
    out = []
    for message in messages:
        if message.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": text_of(message.content),
            }
            role, content = "user", [result | ({"is_error": True} if message.is_error else {})]
        elif message.role == "assistant" and message.reasoning:
            role, content = "assistant", message.reasoning
        else:
            role, content = message.role, message.content
        if out and out[-1]["role"] == role:
            out[-1]["content"] = blocks_of(out[-1]["content"]) + blocks_of(content)
        else:
            out.append({"role": role, "content": content})
    return out


def text_of(content):
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


def blocks_of(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
