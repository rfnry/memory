class MemoryException(Exception):
    pass


class ConfigurationError(MemoryException):
    pass


class InputError(MemoryException):
    pass


class ModelError(MemoryException):
    pass


class StorageError(MemoryException):
    pass


class NotFound(MemoryException):
    pass
