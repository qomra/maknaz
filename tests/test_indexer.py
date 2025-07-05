import os
from maknaz.index import Indexer
from maknaz.config import LOCAL_MAKNAZ_DIR

# get full path MAKNAZ_MODULES_CACHE
HUB = os.environ.get("MAKNAZ_MODULES_CACHE", LOCAL_MAKNAZ_DIR)
def test_indexer():
    index = Indexer(HUB, f"{HUB}/index.json")
    index.index()
    
