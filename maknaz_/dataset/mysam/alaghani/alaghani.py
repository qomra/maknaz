import json
import glob
import datasets
import os
from maknaz.config import LOCAL_MAKNAZ_DIR

# get the path to the hub
HUB = os.environ.get("MAKNAZ_MODULES_CACHE", LOCAL_MAKNAZ_DIR)


class AlAghaniDatasetV1(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="Dataset containing Maysam dataset V1",
            
            features=datasets.Features(
                {
                    "conversation": [{
                        "role": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }]
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.
        
        Here we define the splits and download or extract the data if necessary.
        """
        data_json_train = [f"{HUB}/dataset/mysam/alaghani/alaghani_train.json"]
        data_json_test = [f"{HUB}/dataset/mysam/alaghani/alaghani_test.json"]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": data_json_train},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": data_json_test},
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
        dataset_file = files[0]
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)

        for item in dataset:
            
            """
            {
                "poem_id": "d83d607a973a",
                "prompt": "أريدك أن تكتب قصيدة قصيرة من بيتين فقط على البحر الطويل، تصور حالة رجل يخاطب ساقيه أو خادمه، يطلب منه أن يملأ كأسه بالخمر بعد أن كان يملؤها بالدماء (ربما كناية عن تحوله من حياة الحرب والقتال إلى حياة اللهو والشراب). \n\nالقصيدة يجب أن:\n- تستخدم قافية الراء المفتوحة\n- تحمل طابع الغزل أو الخمريات \n- تتضمن صوراً شعرية تقارن بين الدم والخمر\n- تظهر رقة في المشاعر وحباً للمتعة والسرور\n- تنتهي بطلب عدم الإهمال أو النسيان\n\nاجعل الأسلوب كلاسيكياً أنيقاً يليق بشعر العصر العباسي، مع استخدام ألفاظ رقيقة ومعبرة عن التحول من الجد إلى اللهو.",
                "poem": "أيترع ضخضاحي دما بعد ما غدت ... عليّ به مكنونة مترعا خمرا\nفإن كنت منّي أو تحبّ مسرّتي ... فلا تغفلن قبل الصّباح له كسرا",
                "metadata": {
                    "page": 2531,
                    "context": "وفيه لمالك خفيف ثقيل أوّل بالبنصر عن يونس والهشاميّ.\nبرهان محمد بن موسى المنجم على أنه أحسن الناس غناء:\nأخبرني عليّ بن هارون قال حدّثني عبيد اللّه بن عبد اللّه بن طاهر قال:\nكان محمد بن موسى المنجّم يقول: حكمت أنّ إبراهيم بن المهديّ أحسن الناس كلّهم غناء ببرهان، وذلك أنّي كنت أراه بمجالس الخلفاء مثل المأمون والمعتصم يغني المغنون ويغنّي، فإذا ابتدأ الصوت لم يبق من الغلمان والمتصرّفين في الخدمة وأصحاب الصناعات والمهن الصّغار والكبار أحد إلّا ترك ما في يده وقرب من أقرب موضع يمكنه أن يسمعه، فلا يزال مصغيا إليه لاهيا عمّا كان فيه ما دام يغنّي، حتى إذا أمسك وتغنّى غيره رجعوا إلى التّشاغل بما كانوا فيه ولم يلتفتوا إلى ما يسمعون./ ولا برهان أقوى من هذا في مثل هذا من شهادة الفطن له واتّفاق الطّبائع - مع اختلافها وتشعّب طرقها - على الميل إليه والانقياد له.\nكانت له أشياء لم يكن لأحد مثلها:\nحدّثني أحمد بن جعفر جحظة قال حدّثني هبة اللّه بن إبراهيم بن المهديّ قال:\nقلت للمعتصم: كانت لأبي أشياء لم يكن لأحد/ مثلها. فقال: وما هي؟ قلت: شارية وزامرتها معمعة.\nفقال: أمّا شارية فعندنا، فما فعلت الزّامرة؟ قلت: ماتت. قال: وماذا؟ قلت: وساقيته مكنونة، ولم ير أحسن وجها ولا ألين ولا أظرف منها. قال: فما فعلت؟ قلت: ماتت. قال: وماذا؟ قلت: نخلة كانت تحمل رطبا طول الرّطبة\nقلت: الساعة واللّه حجمني فيه أبو حرملة فسألته أن يهبه لي ففعل، ووجّهت به إلى منزلي فغسل ونظّف وأعيد إلى خزانتي، فرأيت أبي فيما يرى النائم في ليلتي تلك وهو يقول لي:",
                    "analysis": {
                        "عدد_الأبيات": "2",
                        "القافية": "را",
                        "طول_متوسط": "12"
                    }
                }
            }
            make messages out of prompt and poem
            """
          
            messages = [
                {
                    "role": "system",
                    "content": "أنت المساعد الشاعر. تصنع شعرا للمستخدم بناءً على طلباته. يجب أن يكون شعرك فصيحا وأنيقاً، ويستخدم ألفاظاً رقيقة ومعبرة. إذا طلب المستخدم شيئاً محدداً، يجب عليك اتباع التعليمات بدقة."
                },
                {
                    "role": "user",
                    "content": item["prompt"]
                },
                {
                    "role": "assistant",
                    "content": item["poem"]
                }
            ]
            yield key, {
                "conversation": messages
            }
            key += 1
               