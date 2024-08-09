from arhub.client import Client

__all__ = ["Client"]
__version__ = "0.1.0"

def pull(repo_id: str, load: bool = True, download: bool = False, **kwargs):
    client = Client()
    return client.pull(repo_id, load=load, download=download, **kwargs)

def push(data_object, repo_id,api_path=None):
    client = Client()
    return client.push(data_object, repo_id,api_path=api_path)