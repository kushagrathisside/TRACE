import json
import logging
import sys

sys.path.append("/home/kushagra/TRACE/backend")

logging.basicConfig(level=logging.INFO)

from rag.pipeline import run  # noqa: E402
from rag.semantic_cache import SemanticCache  # noqa: E402
from rag.vector_store import VectorStoreManager  # noqa: E402

vs = VectorStoreManager.get_or_create()
# Clear the cache because the DB was just populated
SemanticCache(vs._store._client).invalidate()

print("Starting pipeline run...")
result = run("bipedal locomotion")
print(json.dumps(result, indent=2))
