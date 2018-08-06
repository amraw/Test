from modules.runtime import generate_pipeline
from modules.pipeline_generator.predefined_pipelines.handgeometry_22_pipeline import handgeometry_dataset_pipline
from tests.common import run_pipeline

def test_22_handgeomtry():
    dataset_id = 22
    run_pipeline(dataset_id, handgeometry_dataset_pipline())

test_22_handgeomtry()