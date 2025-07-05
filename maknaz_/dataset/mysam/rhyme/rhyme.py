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
DATA_FILE = os.path.join(HUB, "dataset", "mysam", "alaghani", "poems.jsonl")
RANDOM_SEED = 42
PATTERNS_PER_POEM = 3
RHYME_HINT_PERCENTAGE = 0.75
INCLUDE_ANSWER_IN_HINTS_PERCENTAGE = 0.50
MAX_RHYME_HINTS = 20
MAX_FULL_VERSES = 8  # Limit to 8 full verses (16 hemistichs)

print(f"🎯 Arabic Rhyme Completion Dataset Configuration:")
print(f"📊 Source File: {DATA_FILE}")
print(f"split: 90% train, 10% test (seed={RANDOM_SEED})")
print(f"✨ Prompting Strategy: {RHYME_HINT_PERCENTAGE:.0%} Few-Shot, with answers in hints {INCLUDE_ANSWER_IN_HINTS_PERCENTAGE:.0%} of the time.")
print(f"📏 Max Full Verses: {MAX_FULL_VERSES}")

def reformat_poem_verses(raw_lines: list[str]) -> list[str]:
    """
    Takes a list of hemistichs (half-verses) and joins each pair
    into a single full verse, separated by '...'. Limits to 8 full verses (16 hemistichs).
    """
    # Truncate to 16 hemistichs to ensure max 8 full verses
    raw_lines = [l.strip() for l in raw_lines if l.strip()][:16]
    reformatted_verses = []
    # Iterate over the raw lines in steps of 2, safely handling odd numbers
    for i in range(0, len(raw_lines) - (len(raw_lines) % 2), 2):
        sadr = raw_lines[i].strip()
        ajuz = raw_lines[i+1].strip()
        full_verse = f"{sadr} ... {ajuz}"
        reformatted_verses.append(full_verse)
    return reformatted_verses[:MAX_FULL_VERSES]  # Ensure no more than 8 full verses

def extract_rhyme_ending(word):
    """Extract rhyme ending from Arabic word."""
    clean_word = araby.strip_diacritics(word.strip())
    return ''.join(c for c in clean_word if c.isalpha())

def find_best_consistent_rhyme_segment(verses):
    """
    Finds the best rhyme pattern based on the last word of each full verse.
    """
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
    """
    Pass 1: Process all poems to get basic examples and build a comprehensive rhyme dictionary.
    """
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

def _build_final_examples(potential_examples, rhyme_dictionary):
    """
    Pass 2: Build the final examples with a mix of zero-shot and hint-based prompts.
    """
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

        # Build the task instruction for the user prompt
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

        # Combine task instruction with poem context
        user_content = f"{task_instruction}{hints_text}\n\n{context_text}"

        messages = [
            {"role": "system", "content": "أنت متخصص في الشعر والقافية"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        completion_examples.append({"conversation": messages})

    return completion_examples

class ArabicPoetryRhymeDataset(datasets.GeneratorBasedBuilder):
    """Dataset builder with a two-pass 'Process All, then Split' methodology."""

    _TRAIN_EXAMPLES = []
    _TEST_EXAMPLES = []
    _DATA_PROCESSED = False

    def _info(self):
        return datasets.DatasetInfo(
            description="A dataset for Arabic rhyme completion, with mixed prompts including rhyme-word hints.",
            features=datasets.Features({
                "conversation": [{"role": datasets.Value("string"), "content": datasets.Value("string")}],
            }),
        )

    def _load_and_process_all_data(self):
        """Loads all data, generates all examples, and then splits the final set."""
        if ArabicPoetryRhymeDataset._DATA_PROCESSED:
            return

        print("--- Loading all data from poems.jsonl ---")
        all_poems = []
        if not os.path.exists(DATA_FILE):
             raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    poem_data = json.loads(line)
                    if "error" not in poem_data:
                        all_poems.append(poem_data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
        print(f"Loaded {len(all_poems)} valid poems.")
        
        print("\n--- Pass 1: Generating potential examples and building rhyme dictionary ---")
        potential_examples, rhyme_dict = _generate_potential_examples_and_rhyme_dict(all_poems)
        print(f"✅ Found {len(potential_examples)} potential examples and {len(rhyme_dict)} unique rhymes.")

        print("\n--- Pass 2: Building final examples with enriched prompts ---")
        all_examples = _build_final_examples(potential_examples, rhyme_dict)
        print(f"✅ Generated a total of {len(all_examples)} final examples with mixed prompts.")

        print("\n--- Splitting the FINAL examples into 90% train and 10% test ---")
        train_examples, test_examples = train_test_split(
            all_examples, test_size=0.1, random_state=RANDOM_SEED
        )
        print(f"Split complete: {len(train_examples)} train examples, {len(test_examples)} test examples.")

        # Round down to the nearest 100 for consistency
        print("\n--- Rounding down dataset sizes for consistency ---")
        original_train_size = len(train_examples)
        new_train_size = (original_train_size // 100) * 100
        train_examples = train_examples[:new_train_size]
        print(f"Train examples trimmed from {original_train_size} to {len(train_examples)}")

        original_test_size = len(test_examples)
        new_test_size = (original_test_size // 100) * 100
        test_examples = test_examples[:new_test_size]
        print(f"Test examples trimmed from {original_test_size} to {len(test_examples)}")

        self._TRAIN_EXAMPLES = train_examples
        self._TEST_EXAMPLES = test_examples
        ArabicPoetryRhymeDataset._DATA_PROCESSED = True

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
        """Yields the examples for the pre-split dataset."""
        print(f"\n🎯 Yielding examples for {split_name} split...")
        # limit the number of examples to 1000 for train and 100 for test

        for key, example in enumerate(examples):
            yield key, example
        print(f"🏁 {split_name} dataset complete: {len(examples)} examples yielded.")
        print("-" * 60)