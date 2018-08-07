from modules.pipeline_generator.predefined_pipelines.audio_pipeline import audio_pipeline
from tests.common import run_pipeline

def test_31_urbunsound():
    dataset_id = 31
    run_pipeline(dataset_id, audio_pipeline())

test_31_urbunsound()