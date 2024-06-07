import os
import json
import importlib
from typing import Optional, List, Any
from langchain_core.vectorstores import VectorStore

# get full path ARHUB_MODULES_CACHE
HUB = os.environ.get("ARHUB_MODULES_CACHE", os.path.expanduser("~/.arhub"))
# get absolute path
HUB = os.path.abspath(HUB)

def load_dataset_repo(repo):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is not installed. Please install it using `pip install datasets`")
    # load metadata file from full_path
    with open(os.path.join(repo["full_path"], "metadata.json")) as f:
        metadata = json.load(f)
    
    kind = metadata.get("kind","script")
    if kind == "script":
        return load_dataset(f"{repo['full_path']}",
                data_dir=repo["full_path"],
                trust_remote_code=True,
                download_mode="reuse_cache_if_exists",
                cache_dir=None)
    elif kind == "audio":
        return load_dataset("audiofolder", data_dir=repo["full_path"])



def load_repo(repo):
    if repo["kind"] == "dataset":
        # check if datasets is installed
        return load_dataset_repo(repo)
    elif repo["kind"] == "vectorstore":
        clss_name = repo["class"]
        # import class from string if not already imported
        if clss_name not in globals():
            mod_name = repo["module"]
            try:
                mod = importlib.import_module(mod_name)
                globals()[clss_name] = getattr(mod, clss_name)
            except ImportError:
                raise ImportError(f"Module {mod_name} not found")
        # load vectorstore from full_path
        return globals()[clss_name].load_local(repo["full_path"])
    elif repo["kind"] == "prompt":
        from langchain_core.load.load import loads
        from langchain_core.prompts import BasePromptTemplate
        # load manifest from full_path
        with open(repo["full_path"]) as f:
            res_dict = json.load(f)
        obj = loads(json.dumps(res_dict))
        return obj
        
        

def _get_api_path(api_path: Optional[str]) -> str:
    if api_path is None:
        api_path = HUB
    api_path = os.path.abspath(api_path)
    return api_path

class Client:
    """
    An API Client for NeoHub
    """

    def __init__(self, path: Optional[str] = None):
        self.api_path = _get_api_path(path)
        # load index
        with open(os.path.join(self.api_path, "index.json")) as f:
            self.index = json.load(f)

    def _get_headers(self):
        raise NotImplementedError

    @property
    def _host_url(self) -> str:
        raise NotImplementedError

    def get_settings(self):
        raise NotImplementedError

    def list_repos(self, 
            limit: int = 100, 
            offset: int = 0, 
            kind: Optional[str] = None):
        # load index file @ api_path/index.json
        index = self.index
        repos = index["repos"]
        if kind is not None:
            repos = index["repos"][index[kind][0]:index[kind][1]]
        return [r["full_name"] for r in repos[offset:offset+limit]]
        

    def get_repo(self, repo_full_name: str):
        repo = self.index["repos"][self.index["map"].get(repo_full_name)]
        return repo

    def create_repo(
        self, repo_handle: str, *, description: str = "", is_public: bool = True
    ):
        raise NotImplementedError

    def list_commits(self, repo_full_name: str, limit: int = 100, offset: int = 0):
        raise NotImplementedError

    def like_repo(self, repo_full_name: str):
        raise NotImplementedError

    def unlike_repo(self, repo_full_name: str):
        raise NotImplementedError

    def _get_latest_commit_hash(self, repo_full_name: str) -> Optional[str]:
        raise NotImplementedError

    def update_repo(
        self,
        repo_full_name: str,
        *,
        description: Optional[str] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ):
        raise NotImplementedError

    def push(
        self,
        data_object: Any,
        repo_full_name: str,
        *,
        parent_commit_hash: Optional[str] = "latest",
        new_repo_is_public: bool = False,
        new_repo_description: str = "",
    ):
        # check if data_object is a vectorstore
        if isinstance(data_object, VectorStore):
            # save vectorstore to api_path/repo_full_name
            save_location = os.path.join(self.api_path,"vdb", repo_full_name)
            data_object.save_local(save_location)
            # write metadata to api_path/repo_full_name/metadata.json
            metadata = {
                "kind": "vectorstore",
                "full_path": save_location,
                "class": data_object.__class__.__name__,
                "module": data_object.__module__
            }
            with open(os.path.join(save_location, "metadata.json"), "w") as f:
                json.dump(metadata, f)
        

    def pull(self, owner_repo_commit: str, load: bool = True):
        repo = self.get_repo(owner_repo_commit)
        if load:
            repo = load_repo(repo)
        return repo

