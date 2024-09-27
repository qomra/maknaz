from maknaz.index import Indexer

def test_indexer():
    index = Indexer("./maknaz_/", "./maknaz_/index.json")
    index.index()
    