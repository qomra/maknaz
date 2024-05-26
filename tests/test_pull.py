
def test_pull_script_dataset():
    from neohub import pull
    dataset = pull('aramco/AramcoPub',load=True)
    assert dataset is not None
    assert len(dataset) > 0

def test_pull_audio_dataset():
    from neohub import pull
    dataset = pull('aramco/CommonCommands',load=True)
    assert dataset is not None
    assert len(dataset) > 0