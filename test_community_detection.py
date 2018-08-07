from modules.pipeline_generator.predefined_pipelines.community_detection import community_detection
from tests.common import run_pipeline

def test_70_amazon():
    dataset_id = 70
    run_pipeline(dataset_id, community_detection())

test_70_amazon()