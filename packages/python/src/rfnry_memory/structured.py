from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from rfnry_memory.errors import ModelError
from rfnry_memory.models import GenerationModel, Message, Reply, ToolSpec, Usage, as_reply

OUTPUT_TOOL = "Output"


def _assistant(reply: Reply) -> Message:
    return Message(role="assistant", content=reply.text, tool_calls=reply.tool_calls, reasoning=reply.reasoning)


async def structured_call[T: BaseModel](
    model: GenerationModel,
    *,
    system: str,
    user: str,
    schema: type[T],
    input_schema: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> tuple[T, Usage]:
    tool = ToolSpec(
        name=OUTPUT_TOOL,
        description="Emit the result. Call exactly once; the input is validated against the schema.",
        input_schema=input_schema or schema.model_json_schema(),
    )
    messages: list[Message] = [Message(role="user", content=user)]
    last_error = f"the model never called {OUTPUT_TOOL}"
    usage = Usage()
    for _ in range(max_attempts):
        reply = as_reply(await model.generate(system, messages, [tool]))
        usage = usage + (reply.usage or Usage())
        call = next((c for c in reply.tool_calls if c.name == OUTPUT_TOOL), None)
        feedback = f"call the {OUTPUT_TOOL} tool with the result"
        if call is not None:
            try:
                return schema.model_validate(call.args), usage
            except ValidationError as exc:
                last_error = str(exc)
                feedback = f"invalid; fix and call again: {exc.errors(include_url=False)}"
        if reply.text or reply.tool_calls:
            messages.append(_assistant(reply))
        for made in reply.tool_calls:
            messages.append(Message(role="tool", tool_call_id=made.id, is_error=True, content=feedback))
        if not reply.tool_calls:
            messages.append(Message(role="user", content=f"Call the {OUTPUT_TOOL} tool with the result."))
    raise ModelError(f"structured: {last_error}")
