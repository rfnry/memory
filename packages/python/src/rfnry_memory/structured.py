from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from rfnry_memory.errors import ModelError
from rfnry_memory.models import GenerationModel, Message, ToolSpec, as_reply

OUTPUT_TOOL = "Output"


async def structured_call[T: BaseModel](
    model: GenerationModel,
    *,
    system: str,
    user: str,
    schema: type[T],
    input_schema: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> T:
    tool = ToolSpec(
        name=OUTPUT_TOOL,
        description="Emit the result. Call exactly once; the input is validated against the schema.",
        input_schema=input_schema or schema.model_json_schema(),
    )
    messages: list[Message] = [Message(role="user", content=user)]
    last_error = f"the model never called {OUTPUT_TOOL}"
    for _ in range(max_attempts):
        reply = as_reply(await model.generate(system, messages, [tool]))
        call = next((c for c in reply.tool_calls if c.name == OUTPUT_TOOL), None)
        if call is None:
            if reply.text:
                messages.append(Message(role="assistant", content=reply.text))
            messages.append(Message(role="user", content=f"Call the {OUTPUT_TOOL} tool with the result."))
            continue
        try:
            return schema.model_validate(call.args)
        except ValidationError as exc:
            last_error = str(exc)
            messages.append(
                Message(role="assistant", content=[{"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}])
            )
            messages.append(
                Message(role="tool", tool_call_id=call.id, is_error=True, content=f"invalid; fix and call again: {exc.errors(include_url=False)}")
            )
    raise ModelError(f"structured: {last_error}")
