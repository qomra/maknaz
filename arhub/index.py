import os
import glob
import json
from typing import Optional, List
from .client import _get_api_path

class Indexer:
    def __init__(self, path: Optional[str] = None, output_path: Optional[str] = None):
        # remove last / if exists
        self.api_path = _get_api_path(path).rstrip("/")
        self.output_path = output_path
    
    def _index(self, data:List[str], kind:str,current_index:int=0):
        json_data = []
        map_data = {}
        for d in data:
            # get last two directories
            full_name = "/".join(d.split("/")[-2:])
            owner     = d.split("/")[-2]
            name      = d.split("/")[-1]
            
            full_path = self.api_path + "/"+ kind + "/" + full_name
            json_data.append({
                "full_name":full_name,
                "owner":owner,
                "name":name,
                "kind":kind,
                "full_path":full_path
            })
            real_name = full_name.replace(".json","")
            map_data[real_name] = current_index
            current_index += 1

        return json_data, map_data

    def index(self,return_index=False):
        # list all the repos in the hub
        
        kinds = ["dataset","model","prompt","agent"]
        repos = {"repos":[],"map":{}}
        for kind in kinds:
            kind_repos = glob.glob(f"{self.api_path}/{kind}/*/*")
            indexed_repos,map_data = self._index(kind_repos,kind,len(repos["repos"]))
            repos[kind] = [len(repos["repos"]),len(repos["repos"])+len(indexed_repos)]
            repos["repos"].extend(indexed_repos)
            repos["map"].update(map_data)
            
        if return_index:
            return return_index
        
        with open(self.output_path, "w") as f:
            f.write(json.dumps(repos, indent=4))
        
        
            
        

