import os
from typing import List,Optional,Union
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

HUB = os.environ.get("MAKNAZ_MODULES_CACHE", os.path.expanduser("~/.maknaz"))

class BaseEvaluation(BaseModel):
    model: str
    dataset: str
    split: Optional[str]

class STTEvaluation(BaseEvaluation):
    file_name:  Union[List[List[str]],List[str]]
    actual:     Union[List[List[str]],List[str]]
    prediction: Union[List[List[str]],List[str]]
    wer:        Union[List[List[float]],List[float]]

    def to_pandas(self):
        import pandas as pd
        df = pd.DataFrame({
            "file_name":self.file_name,
            "actual":self.actual,
            "prediction":self.prediction,
            "wer":self.wer
         })
        return df

    @staticmethod
    def load(repo):
        import pandas as pd
        # read ignore bad lines
        df = pd.read_csv(f"{HUB}/{repo['path']}/{repo['split']}.csv",on_bad_lines='warn' )
        #df = pd.read_csv(f"{HUB}/{repo['path']}/{repo['split']}.csv")
        return STTEvaluation(
                model=repo["model"],
                dataset=repo["dataset"],
                file_name=df["file_name"].tolist(),
                actual=df["actual"].tolist(),
                prediction=df["prediction"].tolist(),
                wer=df["wer"].tolist())
    
    def save_local(self,path):
        splits     = [self.split] if self.split else ["train","valid","test"]
        file_name  = self.file_name if self.file_name else [self.file_name]
        actual     = self.actual if type(self.actual[0]) == list else [self.actual]
        prediction = self.prediction if type(self.prediction[0]) == list else [self.prediction]
        wer        = self.wer if type(self.wer[0]) == list else [self.wer]
        for split_name,fileList,alist,plist,werlist in zip(splits,file_name,actual,prediction,wer):
            with open(f"{path}/{split_name}.csv","w") as f:
                f.write("file_name,actual,prediction,wer\n")
                for fn,a,p,e in zip(fileList,alist,plist,werlist):
                    f.write(f"{fn},{a},{p},{e}\n")