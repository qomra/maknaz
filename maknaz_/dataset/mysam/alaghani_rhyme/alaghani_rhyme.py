import json
import datasets
import os
import random
import pyarabic.araby as araby
from sklearn.model_selection import train_test_split
from collections import defaultdict
from maknaz.config import LOCAL_MAKNAZ_DIR

# --- Configuration ---
HUB = os.environ.get("MAKNAZ_MODULES_CACHE", LOCAL_MAKNAZ_DIR)
POEMS_JSONL_FILE = os.path.join(HUB, "dataset", "mysam", "alaghani_rhyme", "poems.jsonl")
ALAGHANI_TRAIN_FILE = os.path.join(HUB, "dataset", "mysam", "alaghani_rhyme", "alaghani_train.json")
ALAGHANI_TEST_FILE = os.path.join(HUB, "dataset", "mysam", "alaghani_rhyme", "alaghani_test.json")

RANDOM_SEED = 42
PATTERNS_PER_POEM = 3
RHYME_HINT_PERCENTAGE = 0.75
INCLUDE_ANSWER_IN_HINTS_PERCENTAGE = 0.50
MAX_RHYME_HINTS = 20
MAX_FULL_VERSES = 8

# Dataset mixing strategy: Use percentage-based sampling relative to generation task size
# RHYME_SAMPLING_PERCENTAGE = 100% means rhyme examples = generation examples
# RHYME_SAMPLING_PERCENTAGE = 50% means rhyme examples = 0.5 * generation examples  
# RHYME_SAMPLING_PERCENTAGE = 200% means rhyme examples = 2 * generation examples
RHYME_SAMPLING_PERCENTAGE = 700  # Percentage of rhyme examples relative to generation task size

print(f"🎯 Combined Arabic Poetry Dataset Configuration:")
print(f"📊 Poems JSONL File: {POEMS_JSONL_FILE}")
print(f"📊 AlAghani Train File: {ALAGHANI_TRAIN_FILE}")
print(f"📊 AlAghani Test File: {ALAGHANI_TEST_FILE}")
print(f"🎵 Rhyme Sampling Percentage: {RHYME_SAMPLING_PERCENTAGE}% (relative to generation task size)")
print(f"🎲 Random Seed: {RANDOM_SEED}")

# Rhyme dataset functions (copied from your original code)
def reformat_poem_verses(raw_lines: list[str]) -> list[str]:
    """Takes a list of hemistichs and joins each pair into a full verse."""
    raw_lines = [l.strip() for l in raw_lines if l.strip()][:16]
    reformatted_verses = []
    for i in range(0, len(raw_lines) - (len(raw_lines) % 2), 2):
        sadr = raw_lines[i].strip()
        ajuz = raw_lines[i+1].strip()
        full_verse = f"{sadr} ... {ajuz}"
        reformatted_verses.append(full_verse)
    return reformatted_verses[:MAX_FULL_VERSES]

def extract_rhyme_ending(word):
    """Extract rhyme ending from Arabic word."""
    clean_word = araby.strip_diacritics(word.strip())
    return ''.join(c for c in clean_word if c.isalpha())

def find_best_consistent_rhyme_segment(verses):
    """Finds the best rhyme pattern based on the last word of each full verse."""
    if len(verses) < 2:
        return [], [], ""
    
    last_words = [extract_rhyme_ending(verse.split()[-1]) for verse in verses if verse.split()]
    if not last_words: return [], [], ""
    
    first_word = last_words[0]
    if len(first_word) < 1: return [], [], ""
    
    best_rhyme, best_consistent_verses, best_indices = "", [], []
    for rhyme_len in [3, 2, 1]:
        if len(first_word) >= rhyme_len:
            rhyme_pattern = first_word[-rhyme_len:]
            consistent_verses = []
            consistent_indices = []
            for i, (verse, word) in enumerate(zip(verses, last_words)):
                if len(word) >= rhyme_len and word[-rhyme_len:] == rhyme_pattern:
                    consistent_verses.append(verse)
                    consistent_indices.append(i)
                else: break
            if len(consistent_verses) > len(best_consistent_verses):
                best_rhyme = rhyme_pattern
                best_consistent_verses = consistent_verses
                best_indices = consistent_indices
    
    if len(best_consistent_verses) < 2: return [], [], ""
    return best_consistent_verses, best_indices, best_rhyme

def _generate_potential_examples_and_rhyme_dict(poems_data):
    """Process all poems to get rhyme completion examples and build rhyme dictionary."""
    potential_examples = []
    rhyme_dictionary = defaultdict(set)
    
    for item_idx, item in enumerate(poems_data):
        poem_text = item.get("poem")
        if not poem_text: continue
        
        raw_lines = [l.strip() for l in poem_text.split('\n') if l.strip()]
        full_verses = reformat_poem_verses(raw_lines)
        
        if len(full_verses) <= 1: continue
        
        consistent_verses, consistent_indices, detected_rhyme = find_best_consistent_rhyme_segment(full_verses)
        if len(consistent_verses) < 2 or not detected_rhyme: continue

        num_consistent_verses = len(consistent_verses)
        num_patterns_to_generate = 1
        if num_consistent_verses > 4:
            num_patterns_to_generate = PATTERNS_PER_POEM

        generated_mask_sets = set()
        for _ in range(num_patterns_to_generate * 2):
            if len(generated_mask_sets) >= num_patterns_to_generate: break
                
            mask_indices_in_consistent_block = []
            if num_consistent_verses == 2:
                mask_indices_in_consistent_block = [1]
            elif num_consistent_verses in [3, 4]:
                mask_indices_in_consistent_block = sorted(list(set([1, num_consistent_verses - 1])))
            elif num_consistent_verses > 4:
                mask_indices_in_consistent_block = sorted(random.sample(range(num_consistent_verses), 3))
            
            mask_set = tuple(mask_indices_in_consistent_block)
            if not mask_set or mask_set in generated_mask_sets: continue
            generated_mask_sets.add(mask_set)

            mask_indices_in_full_poem = [consistent_indices[i] for i in mask_indices_in_consistent_block]

            full_poem_masked_lines = []
            target_words = []
            for i, verse in enumerate(full_verses):
                if i in mask_indices_in_full_poem:
                    words = verse.split()
                    if len(words) >= 1:
                        full_poem_masked_lines.append(' '.join(words[:-1]) + " ___")
                        target_words.append(words[-1])
                    else:
                        full_poem_masked_lines.append(verse)
                else:
                    full_poem_masked_lines.append(verse)
            
            if not target_words: continue

            potential_examples.append({
                "context_text": '\n'.join(full_poem_masked_lines),
                "target_words": target_words,
                "rhyme": detected_rhyme
            })
            for word in target_words:
                rhyme_dictionary[detected_rhyme].add(word)
                
    return potential_examples, rhyme_dictionary

def _build_rhyme_examples(potential_examples, rhyme_dictionary):
    """Build the final rhyme completion examples with hints."""
    completion_examples = []
    for target_example in potential_examples:
        context_text = target_example['context_text']
        target_words = target_example['target_words']
        detected_rhyme = target_example['rhyme']

        if len(target_words) == 1:
            response = target_words[0]
            task_desc = "أكمل الكلمة المفقودة"
            response_format = "اذكر الكلمة فقط."
        else:
            response = ' | '.join(target_words)
            task_desc = f"أكمل الكلمات المفقودة ({len(target_words)} كلمات)"
            response_format = "اذكر الكلمات بنفس ترتيب الأبيات مفصولة بـ |"

        task_instruction = f"{task_desc} بحيث تحافظ على القافية '{detected_rhyme}'. {response_format}"
        
        # Add hints if applicable
        hints_text = ""
        if random.random() < RHYME_HINT_PERCENTAGE:
            all_hint_words = rhyme_dictionary.get(detected_rhyme, set())
            
            if random.random() > INCLUDE_ANSWER_IN_HINTS_PERCENTAGE:
                possible_hints = list(all_hint_words - set(target_words))
            else:
                possible_hints = list(all_hint_words)
            
            if possible_hints:
                num_hints = min(MAX_RHYME_HINTS, len(possible_hints))
                selected_hints = random.sample(possible_hints, num_hints)
                hint_string = "، ".join(selected_hints)
                hints_text = f"\nملاحظة: كلمات أخرى تنتهي بنفس الحرف تشمل: {hint_string}."

        user_content = f"{task_instruction}{hints_text}\n\n{context_text}"

        messages = [
            {"role": "system", "content": "أنت متخصص في الشعر والقافية"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        completion_examples.append({
            "conversation": messages,
            "task_type": "rhyme_completion"
        })

    return completion_examples


class CombinedArabicPoetryDataset(datasets.GeneratorBasedBuilder):
    """Combined dataset for Arabic poetry generation and rhyme completion."""

    _TRAIN_EXAMPLES = []
    _TEST_EXAMPLES = []
    _DATA_PROCESSED = False

    def _info(self):
        return datasets.DatasetInfo(
            description="Combined dataset for Arabic poetry generation and rhyme completion training.",
            features=datasets.Features({
                "conversation": [{"role": datasets.Value("string"), "content": datasets.Value("string")}],
                "task_type": datasets.Value("string"),
            }),
        )

    def _load_poetry_generation_data(self):
        """Load poetry generation examples from pre-split alaghani files."""
        train_examples = []
        test_examples = []
        
        # Load training data
        if os.path.exists(ALAGHANI_TRAIN_FILE):
            with open(ALAGHANI_TRAIN_FILE, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
                for item in train_data:
                    messages = [
                        {
                            "role": "system",
                            "content": "أنت المساعد الشاعر. تصنع شعرا للمستخدم بناءً على طلباته. يجب أن يكون شعرك فصيحاً وأنيقاً، ويستخدم ألفاظاً رقيقة ومعبرة. إذا طلب المستخدم شيئاً محدداً، يجب عليك اتباع التعليمات بدقة."
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
                    train_examples.append({
                        "conversation": messages,
                        "task_type": "poetry_generation"
                    })
        
        # Load test data
        if os.path.exists(ALAGHANI_TEST_FILE):
            with open(ALAGHANI_TEST_FILE, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                for item in test_data:
                    messages = [
                        {
                            "role": "system",
                            "content": "أنت المساعد الشاعر. تصنع شعرا للمستخدم بناءً على طلباته. يجب أن يكون شعرك فصيحاً وأنيقاً، ويستخدم ألفاظاً رقيقة ومعبرة. إذا طلب المستخدم شيئاً محدداً، يجب عليك اتباع التعليمات بدقة."
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
                    test_examples.append({
                        "conversation": messages,
                        "task_type": "poetry_generation"
                    })
        
        print(f"✅ Loaded {len(train_examples)} poetry generation train examples")
        print(f"✅ Loaded {len(test_examples)} poetry generation test examples")
        return train_examples, test_examples

    def _load_rhyme_completion_data(self):
        """Load and generate rhyme completion examples from poems.jsonl."""
        print("--- Loading all poems from poems.jsonl ---")
        all_poems = []
        if not os.path.exists(POEMS_JSONL_FILE):
            raise FileNotFoundError(f"Dataset file not found: {POEMS_JSONL_FILE}")
        
        with open(POEMS_JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    poem_data = json.loads(line)
                    if "error" not in poem_data:
                        all_poems.append(poem_data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
        
        print(f"Loaded {len(all_poems)} valid poems.")
        
        print("--- Pass 1: Generating potential examples and building rhyme dictionary ---")
        potential_examples, rhyme_dict = _generate_potential_examples_and_rhyme_dict(all_poems)
        print(f"✅ Found {len(potential_examples)} potential rhyme examples and {len(rhyme_dict)} unique rhymes.")

        print("--- Pass 2: Building final rhyme completion examples ---")
        all_rhyme_examples = _build_rhyme_examples(potential_examples, rhyme_dict)
        print(f"✅ Generated a total of {len(all_rhyme_examples)} rhyme completion examples.")

        # Split rhyme examples into train/test (90/10)
        train_rhyme, test_rhyme = train_test_split(
            all_rhyme_examples, test_size=0.1, random_state=RANDOM_SEED
        )
        
        print(f"✅ Split rhyme data: {len(train_rhyme)} train, {len(test_rhyme)} test")
        return train_rhyme, test_rhyme

    def _create_balanced_combined_dataset(self, poetry_train, poetry_test, rhyme_train, rhyme_test):
        """Combine datasets using percentage-based sampling relative to generation task size."""
        
        # Calculate target rhyme examples based on percentage of generation task size
        total_poetry = len(poetry_train) + len(poetry_test)
        target_rhyme_total = int(total_poetry * RHYME_SAMPLING_PERCENTAGE / 100)
        
        print(f"📊 Dataset balancing based on {RHYME_SAMPLING_PERCENTAGE}% sampling:")
        print(f"   Total Poetry Examples: {total_poetry}")
        print(f"   Target Rhyme Examples: {target_rhyme_total} ({RHYME_SAMPLING_PERCENTAGE}% of poetry size)")
        print(f"   Available Rhyme Examples: {len(rhyme_train) + len(rhyme_test)}")
        
        # Calculate target splits maintaining proportions
        poetry_train_ratio = len(poetry_train) / total_poetry
        poetry_test_ratio = len(poetry_test) / total_poetry
        
        target_rhyme_train = int(target_rhyme_total * poetry_train_ratio)
        target_rhyme_test = target_rhyme_total - target_rhyme_train
        
        # Sample rhyme examples to match target
        random.seed(RANDOM_SEED)
        if target_rhyme_train <= len(rhyme_train):
            sampled_rhyme_train = random.sample(rhyme_train, target_rhyme_train)
        else:
            # If we need more than available, use all available and warn
            sampled_rhyme_train = rhyme_train
            print(f"⚠️  Warning: Requested {target_rhyme_train} train rhyme examples, but only {len(rhyme_train)} available")
        
        if target_rhyme_test <= len(rhyme_test):
            sampled_rhyme_test = random.sample(rhyme_test, target_rhyme_test)
        else:
            # If we need more than available, use all available and warn
            sampled_rhyme_test = rhyme_test
            print(f"⚠️  Warning: Requested {target_rhyme_test} test rhyme examples, but only {len(rhyme_test)} available")
        
        # Combine train and test sets
        final_train = poetry_train + sampled_rhyme_train
        final_test = poetry_test + sampled_rhyme_test
        
        # Shuffle both sets
        random.seed(RANDOM_SEED)
        random.shuffle(final_train)
        random.shuffle(final_test)
        
        # Calculate actual percentages for reporting
        actual_rhyme_percentage = (len(sampled_rhyme_train) + len(sampled_rhyme_test)) / total_poetry * 100
        
        print(f"📊 Final combined dataset:")
        print(f"   Train: {len(final_train)} examples")
        print(f"     - Poetry: {len(poetry_train)} ({len(poetry_train)/len(final_train):.1%})")
        print(f"     - Rhyme: {len(sampled_rhyme_train)} ({len(sampled_rhyme_train)/len(final_train):.1%})")
        print(f"   Test: {len(final_test)} examples")
        print(f"     - Poetry: {len(poetry_test)} ({len(poetry_test)/len(final_test):.1%})")
        print(f"     - Rhyme: {len(sampled_rhyme_test)} ({len(sampled_rhyme_test)/len(final_test):.1%})")
        print(f"   Actual Rhyme Sampling: {actual_rhyme_percentage:.1f}% of poetry task size")
        
        return final_train, final_test

    def _load_and_process_all_data(self):
        """Load all data, combine, and create final splits."""
        if CombinedArabicPoetryDataset._DATA_PROCESSED:
            return

        print("🔄 Loading poetry generation data...")
        poetry_train, poetry_test = self._load_poetry_generation_data()
        
        print("🔄 Loading and generating rhyme completion data...")
        rhyme_train, rhyme_test = self._load_rhyme_completion_data()
        
        print("🔄 Creating balanced combined dataset...")
        final_train, final_test = self._create_balanced_combined_dataset(
            poetry_train, poetry_test, rhyme_train, rhyme_test
        )
        
        self._TRAIN_EXAMPLES = final_train
        self._TEST_EXAMPLES = final_test
        CombinedArabicPoetryDataset._DATA_PROCESSED = True

    def _split_generators(self, dl_manager):
        self._load_and_process_all_data()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"examples": self._TRAIN_EXAMPLES, "split_name": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"examples": self._TEST_EXAMPLES, "split_name": "test"},
            ),
        ]

    def _generate_examples(self, examples, split_name):
        """Yield the examples for the dataset."""
        print(f"\n🎯 Yielding examples for {split_name} split...")
        for key, example in enumerate(examples):
            yield key, example
        print(f"🏁 {split_name} dataset complete: {len(examples)} examples yielded.")
        print("-" * 60)