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
DATA_FILE = os.path.join(HUB, "dataset", "mysam", "poem_completion", "poems.jsonl")
RANDOM_SEED = 42
MAX_FULL_VERSES = 8  # Limit to 8 full verses (16 hemistichs)

# Bridging task configuration
COMPLETION_PERCENTAGE = 30  # Give 30% of poem, ask to complete the rest
MIN_POEM_LENGTH = 4  # Minimum poem length to be used for completion task

print(f"🎯 Arabic Poetry Bridging Dataset Configuration:")
print(f"📊 Source File: {DATA_FILE}")
print(f"🔗 Bridging Task: Poem Completion ({COMPLETION_PERCENTAGE}% given, complete the rest)")
print(f"📏 Min Poem Length: {MIN_POEM_LENGTH} verses")
print(f"🎲 Random Seed: {RANDOM_SEED}")

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

def detect_meter_from_metadata(item):
    """Extract meter information from poem metadata."""
    meter = item.get("meter", "")
    if meter:
        return meter
    return "البحر غير محدد"

def detect_rhyme_pattern(verses):
    """
    Detect rhyme pattern from the poem verses.
    Returns a rhyme description in the style of the prompted dataset.
    """
    if len(verses) < 2:
        return "القافية غير محددة"
    
    # Get last words from verses
    last_words = []
    for verse in verses:
        words = verse.split()
        if words:
            last_word = extract_rhyme_ending(words[-1])
            last_words.append(last_word)
    
    if not last_words:
        return "القافية غير محددة"
    
    # Find common ending pattern
    first_word = last_words[0]
    if len(first_word) < 1:
        return "القافية غير محددة"
    
    # Try different rhyme lengths (3, 2, 1 characters)
    for rhyme_len in [3, 2, 1]:
        if len(first_word) >= rhyme_len:
            rhyme_pattern = first_word[-rhyme_len:]
            # Check if most words follow this pattern
            matching_count = sum(1 for word in last_words 
                               if len(word) >= rhyme_len and word[-rhyme_len:] == rhyme_pattern)
            if matching_count >= len(last_words) * 0.7:  # At least 70% match
                # Convert to Arabic description format
                return f"{rhyme_pattern}"
    
    return "القافية متنوعة"

def generate_user_prompts():
    """Generate variety of user prompt templates for poem completion."""
    prompts = [
        "أريدك أن تكمل هذه القصيدة لتصبح {total_verses} أبيات على {meter} بقافية '{rhyme}':",
        "اكتب تكملة هذه القصيدة ({remaining_verses} أبيات إضافية) على {meter} بقافية '{rhyme}':",
        "أريد منك أن تكتب {remaining_verses} أبيات أخرى على {meter} بقافية '{rhyme}' لإكمال:",
        "أكمل هذه القصيدة بإضافة {remaining_verses} أبيات آخرين على {meter} بقافية '{rhyme}':",
        "اكتب لي المزيد من الأبيات (حوالي {remaining_verses} أبيات) على نفس الوزن والقافية:",
        "أريدك أن تُطول هذه القصيدة لتصبح {total_verses} أبيات تقريباً على {meter} بقافية '{rhyme}':"
    ]
    return prompts

def _generate_completion_examples(poems_data):
    """
    Generate poem completion examples from the 100K dataset.
    """
    completion_examples = []
    prompt_templates = generate_user_prompts()
    
    for item_idx, item in enumerate(poems_data):
        poem_text = item.get("poem")
        if not poem_text: 
            continue
        
        raw_lines = [l.strip() for l in poem_text.split('\n') if l.strip()]
        full_verses = reformat_poem_verses(raw_lines)
        
        # Skip short poems
        if len(full_verses) < MIN_POEM_LENGTH:
            continue
        
        # Extract metadata
        meter = detect_meter_from_metadata(item)
        rhyme = detect_rhyme_pattern(full_verses)
        
        # Calculate how many verses to give vs complete
        total_verses = len(full_verses)
        given_verses_count = max(1, int(total_verses * COMPLETION_PERCENTAGE / 100))
        remaining_verses_count = total_verses - given_verses_count
        
        if remaining_verses_count < 1:
            continue
        
        # Split poem into given part and completion part
        given_verses = full_verses[:given_verses_count]
        completion_verses = full_verses[given_verses_count:]
        
        # Create user prompt
        prompt_template = random.choice(prompt_templates)
        user_prompt = prompt_template.format(
            total_verses=total_verses,
            remaining_verses=remaining_verses_count,
            meter=meter,
            rhyme=rhyme
        )
        
        # Add the given verses to the prompt
        given_poem_text = '\n'.join(given_verses)
        user_content = f"{user_prompt}\n{given_poem_text}"
        
        # Create completion response
        completion_text = '\n'.join(completion_verses)
        
        # Create conversation format
        messages = [
            {"role": "system", "content": "أنت شاعر يكمل القصائد بناءً على المتطلبات المحددة"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion_text}
        ]
        
        completion_examples.append({"conversation": messages})
        
        # Optional: Create variations with different split points for longer poems
        if total_verses >= 6:
            # Try giving 50% instead of 30%
            alt_given_count = max(2, int(total_verses * 0.5))
            alt_remaining_count = total_verses - alt_given_count
            
            if alt_remaining_count >= 2:
                alt_given_verses = full_verses[:alt_given_count]
                alt_completion_verses = full_verses[alt_given_count:]
                
                alt_prompt_template = random.choice(prompt_templates)
                alt_user_prompt = alt_prompt_template.format(
                    total_verses=total_verses,
                    remaining_verses=alt_remaining_count,
                    meter=meter,
                    rhyme=rhyme
                )
                
                alt_given_poem_text = '\n'.join(alt_given_verses)
                alt_user_content = f"{alt_user_prompt}\n{alt_given_poem_text}"
                alt_completion_text = '\n'.join(alt_completion_verses)
                
                alt_messages = [
                    {"role": "system", "content": "أنت شاعر يكمل القصائد بناءً على المتطلبات المحددة"},
                    {"role": "user", "content": alt_user_content},
                    {"role": "assistant", "content": alt_completion_text}
                ]
                
                completion_examples.append({"conversation": alt_messages})
                
    return completion_examples

class ArabicPoetryBridgingDataset(datasets.GeneratorBasedBuilder):
    """Dataset builder for the bridging task: poem completion with constraints."""

    _TRAIN_EXAMPLES = []
    _TEST_EXAMPLES = []
    _DATA_PROCESSED = False

    def _info(self):
        return datasets.DatasetInfo(
            description="A bridging dataset for Arabic poem completion with meter and rhyme constraints.",
            features=datasets.Features({
                "conversation": [{"role": datasets.Value("string"), "content": datasets.Value("string")}],
            }),
        )

    def _load_and_process_all_data(self):
        """Loads all data, generates completion examples, and splits the final set."""
        if ArabicPoetryBridgingDataset._DATA_PROCESSED:
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
        
        print("\n--- Generating poem completion examples ---")
        all_examples = _generate_completion_examples(all_poems)
        print(f"✅ Generated a total of {len(all_examples)} completion examples.")

        if len(all_examples) == 0:
            raise ValueError("No valid completion examples generated!")

        print("\n--- Splitting the examples into 90% train and 10% test ---")
        train_examples, test_examples = train_test_split(
            all_examples, test_size=0.1, random_state=RANDOM_SEED
        )
        print(f"Split complete: {len(train_examples)} train examples, {len(test_examples)} test examples.")

        # Round down to the nearest 100 for consistency
        print("\n--- Rounding down dataset sizes for consistency ---")
        original_train_size = len(train_examples)
        new_train_size = (original_train_size // 100) * 100
        train_examples = train_examples[:new_train_size] if new_train_size > 0 else train_examples[:100]
        print(f"Train examples trimmed from {original_train_size} to {len(train_examples)}")

        original_test_size = len(test_examples)
        new_test_size = (original_test_size // 100) * 100
        test_examples = test_examples[:new_test_size] if new_test_size > 0 else test_examples[:10]
        print(f"Test examples trimmed from {original_test_size} to {len(test_examples)}")

        self._TRAIN_EXAMPLES = train_examples
        self._TEST_EXAMPLES = test_examples
        ArabicPoetryBridgingDataset._DATA_PROCESSED = True

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
        
        for key, example in enumerate(examples):
            yield key, example
        print(f"🏁 {split_name} dataset complete: {len(examples)} examples yielded.")
        print("-" * 60)