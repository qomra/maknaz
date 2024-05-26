from neohub.client import Client

__all__ = ["Client"]
__version__ = "0.1.0"

def pull(repo_id: str, load: bool = True):
    client = Client()
    return client.pull(repo_id, load=load)

def push(data_object, repo_id):
    client = Client()
    return client.push(data_object, repo_id)