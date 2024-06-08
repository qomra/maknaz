import json
import glob
import datasets
import os

# get the path to the hub
HUB = os.environ.get("ARHUB_MODULES_CACHE", os.path.expanduser("~/.arhub"))


class ArticleSummaryDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="Dataset containing articles with their summaries.",
            features=datasets.Features(
                {
                    "summary": datasets.Value("string"),
                    "article_summary": datasets.Value("string"),
                    "article_name": datasets.Value("string"),
                    "name": datasets.Value("string"),
                    "lawStatus": datasets.Value("string"),
                    "issueDateGregorian": datasets.Value("string"),
                    "issueDateUmAlqura": datasets.Value("string"),
                    "publishDateGregorian": datasets.Value("string"),
                    "publishDateUmAlqura": datasets.Value("string")
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.
        
        Here we define the splits and download or extract the data if necessary.
        """
        json_files = sorted(glob.glob(f"{HUB}/dataset/wali/laws/*.json"))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": json_files},
            ),
        ]

    def _generate_examples(self, files):
        """
        Yields examples as (key, example) tuples.
        
        Args:
            files (list): List of JSON files.
        
        Yields:
            Tuple[int, dict]: The key and the dictionary of summary.
        """
        key = 0
        in_columns = [
            "summary",
            "name",
            "lawStatus",
            "issueDateGregorian",
            "issueDateUmAlqura",
            "publishDateGregorian",
            "publishDateUmAlqura",
        ]
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                # check if value and articles are in the json file otherwise skip
                if "value" in data and "articles" in data["value"] and len(data["value"]["articles"]) > 0:
                    for index,article in enumerate(data["value"]["articles"]):
                        summary = article.get("summary")
                        item = {"article_summary": summary, "article_name": index}
                        for column in in_columns:
                            item[column] = data["value"].get(column,None)
                        if summary:
                            yield key, item
                            key += 1
                        else:
                            continue
                else:
                    continue