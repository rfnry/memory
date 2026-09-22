from rfnry_memory.errors import (
    ConfigurationError,
    InputError,
    MemoryException,
    ModelError,
    NotFoundError,
    StorageError,
)


def test_every_error_is_a_memory_exception_and_none_shadows_a_builtin() -> None:
    for cls in (ConfigurationError, InputError, ModelError, StorageError, NotFoundError):
        assert issubclass(cls, MemoryException)
        assert issubclass(cls, Exception)
    assert MemoryError not in MemoryException.__mro__
