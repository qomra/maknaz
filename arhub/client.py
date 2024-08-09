import os
import json
import shutil
import importlib
from typing import Optional, List, Any
from langchain_core.vectorstores import VectorStore

from .types import *

# get script path
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
# get full path ARHUB_MODULES_CACHE
HUB = os.environ.get("ARHUB_MODULES_CACHE", f"{SCRIPT_PATH}/../hub")
# get absolute path
HUB = os.path.abspath(HUB)

def load_dataset_repo(repo,split=None,**kwargs):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is not installed. Please install it using `pip install datasets`")

    
    kind = repo.get("loader","script")
    path = f"{HUB}/{repo['path']}"
    if kind == "script":
        return load_dataset(path,
                data_dir=path,
                trust_remote_code=True,
                download_mode="reuse_cache_if_exists",
                cache_dir=None)
    elif kind == "audio":
        if split is None:
            return load_dataset("audiofolder", data_dir=path,split=split)
        
        return {split:load_dataset("audiofolder", data_dir=path,split=split)}

def load_repo(repo, **kwargs):
    if repo["kind"] == "dataset":
        # check if datasets is installed
        return load_dataset_repo(repo,**kwargs)
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
        # load vectorstore from path
        return globals()[clss_name].load_local(f"{HUB}/{repo['path']}")
    
    elif repo["kind"] == "prompt":
        from langchain_core.load.load import loads
        from langchain_core.prompts import BasePromptTemplate
        # load manifest from path
        with open(f"{HUB}/{repo['path']}") as f:
            res_dict = json.load(f)
        obj = loads(json.dumps(res_dict))
        return obj
    
    elif repo["kind"] == "evaluation":
        evaluation = globals()[repo["class"]].load(repo)
        return evaluation

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
        

    def get_repo(self, repo_full_name: str, download: bool = False):
        repo = None
        # check if repo exists in index and snapshot exists
        if repo_full_name in self.index["map"] :
            repo = self.index["repos"][self.index["map"].get(repo_full_name)]
            path = f"{self.api_path}/{repo['path']}"
            if not download or \
                repo["kind"] != "model" or \
                os.path.exists(f"{path}/snapshots") or \
                os.path.exists(f"{path}/model.safetensors"):
                return repo
        # check if it is a model in huggingface hub
        if not repo and not download:
            print("Repo doesn't exist in arhub index. Use download=True to download the repo from huggingface hub")
            return None
        
        # download repo from huggingface hub
        from huggingface_hub import snapshot_download,HfApi
        hf_api = HfApi()
        repo_type = "model"
        repo_kind = "model"
        repo_author = repo_full_name.split("/")[0]
        repo_name = repo_full_name.split("/")[1]
        try:
            repo = hf_api.repo_info(repo_full_name)
        except:
            try:
                repo_type = "dataset"
                repo_kind = "text"
                repo = hf_api.dataset_info(repo_full_name,repo_type="dataset")
                repo_tags = repo["tags"]
                # find task_categories:category in tags
                for tag in repo_tags:
                    if "task_categories" in tag:
                        repo_kind = tag.split(":")[1]
                        break
                
            except:
                print(f"Repo {repo_full_name} not found in huggingface hub")
                return None
        
        repo_path = os.path.join(repo_type,repo_full_name)
        repo = snapshot_download(repo_full_name, cache_dir=f"{self.api_path}/{repo_type}", repo_type=repo_type, ignore_patterns=["*.msgpack", "*.h5","*.bin"])
        if repo_type == "model":
            # mv folder from models--author--repo to author/repo
            current_path = f"{self.api_path}/{repo_type}/models--{repo_author}--{repo_name}"
            new_path = f"{self.api_path}/{repo_type}/{repo_author}/{repo_name}"
            print(f"Moving {current_path} to {new_path}")
            # make author directory
            os.makedirs(os.path.join(self.api_path,repo_type,repo_author),exist_ok=True)
            # copy directory to new path
            shutil.copytree(current_path,new_path, dirs_exist_ok=True)
 
    
        repo = {
            'full_name': repo_full_name,
            'owner': repo_author, 
            'name': repo_name, 
            'kind': repo_type, 'path': repo_path}
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
        api_path: Optional[str] = None,
        *,
        parent_commit_hash: Optional[str] = "latest",
        new_repo_is_public: bool = False,
        new_repo_description: str = "",
    ):
        if api_path is None:
            api_path = self.api_path

        # check if data_object is a vectorstore
        if isinstance(data_object, VectorStore):
            # save vectorstore to api_path/repo_full_name
            save_location = os.path.join(api_path,"vdb", repo_full_name)
            os.makedirs(save_location,exist_ok=True)
            data_object.save_local(save_location)
            # write info to api_path/repo_full_name/info.json
            info = {
                "class": data_object.__class__.__name__,
                "module": data_object.__module__
            }
            with open(os.path.join(save_location, "info.json"), "w") as f:
                json.dump(info, f,indent=4)
        
        # check if data_object is evaluation
        if isinstance(data_object,STTEvaluation):
            save_location = os.path.join(api_path,"evaluation", repo_full_name)
            # make directory if not exists
            os.makedirs(save_location,exist_ok=True)
            data_object.save_local(save_location)
            # write info to api_path/repo_full_name/info.json
            info = {
                "class": data_object.__class__.__name__,
                "module": data_object.__module__,
                "model": data_object.model,
                "dataset": data_object.dataset,
                "split": data_object.split
            }
            with open(os.path.join(save_location, "info.json"), "w") as f:
                json.dump(info, f,indent=4)

        # check if data_object is a finetunedmodel
        if isinstance(data_object,ACFinetunedModel):
            current_local_path = data_object.local_path
            new_path           = os.path.join(api_path,"model",repo_full_name)
            owner              = repo_full_name.split("/")[0]
            os.makedirs(os.path.join(api_path,"model",owner),exist_ok=True)
            os.rename(current_local_path,new_path)
            info = {
                "base_model": data_object.base_model,
                "dataset": data_object.dataset
            }
            with open(os.path.join(new_path, "info.json"), "w") as f:
                json.dump(info, f,indent=4)

    def pull(self, owner_repo_commit: str, load: bool = True, download: bool = False, **kwargs):

        repo = self.get_repo(owner_repo_commit,download=download)
        if repo is None:
            print(f"Repo {owner_repo_commit} not found")
            return None
        if load:
            repo = load_repo(repo, **kwargs)
        return repo

