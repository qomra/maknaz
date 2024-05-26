from neohub.index import Indexer

def test_indexer():
    index = Indexer("./hub/", "./hub/index.json")
    index.index()
    