from typing import List,Optional,Union
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator


class BaseEvaluation(BaseModel):
    model: str
    dataset: str
    split: Optional[str]

class STTEvaluation(BaseEvaluation):
    file_name:  Union[List[List[str]],List[str]]
    actual:     Union[List[List[str]],List[str]]
    prediction: Union[List[List[str]],List[str]]
    wer:        Union[List[List[float]],List[float]]
    
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