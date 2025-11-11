from cosmos_predict2._src.imaginaire.utils.easy_io.backends.base_backend import BaseStorageBackend
from cosmos_predict2._src.imaginaire.utils.easy_io.backends.boto3_backend import Boto3Backend
from cosmos_predict2._src.imaginaire.utils.easy_io.backends.http_backend import HTTPBackend
from cosmos_predict2._src.imaginaire.utils.easy_io.backends.local_backend import LocalBackend
from cosmos_predict2._src.imaginaire.utils.easy_io.backends.registry_utils import (
    backends,
    prefix_to_backends,
    register_backend,
)

# MSC (Multi Storage Client) backend is optional and depends on the external
# multistorageclient package. Some environments—such as local development or
# air-gapped deployments—do not have this dependency installed. Importing it
# unconditionally would raise a ModuleNotFoundError during module import time,
# preventing the rest of easy_io (local/http/boto backends) from working.  We
# therefore try to import it lazily and simply skip registration if it is not
# available.
try:
    from cosmos_predict2._src.imaginaire.utils.easy_io.backends.msc_backend import MSCBackend  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    MSCBackend = None

__all__ = [
    "BaseStorageBackend",
    "LocalBackend",
    "HTTPBackend",
    "Boto3Backend",
    "register_backend",
    "backends",
    "prefix_to_backends",
]

if MSCBackend is not None:
    __all__.append("MSCBackend")
