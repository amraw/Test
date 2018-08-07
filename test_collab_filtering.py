from modules.pipeline_generator.predefined_pipelines.collaborative_filtering_link_prediction import collaborative_filtering_link_prediction
from tests.common import run_pipeline

def test_60_jester():
    dataset_id = 60
    run_pipeline(dataset_id, collaborative_filtering_link_prediction())

test_60_jester()