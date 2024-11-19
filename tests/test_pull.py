from maknaz import pull
def test_pull_script_dataset():
    dataset = pull('wali/laws',load=True)
    assert dataset is not None
    assert len(dataset) > 0

def test_pull_audio_dataset():
    dataset = pull('thamaniya/EarlyVoice',load=True)
    assert dataset is not None
    assert len(dataset) > 0

def test_pull_clip_model():
    model,processor = pull('openai/clip-vit-base-patch32',load=True)
    assert model is not None
    assert processor is not None