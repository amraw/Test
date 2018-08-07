from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PipelineContext
import d3m
import d3m.primitives
import d3m.primitives.datasets
import d3m.primitives.sri.psl
import d3m.primitives.sri
import d3m.primitives.sri.graph
from modules.pipeline_generator.utils.hyperparams import add_hyperparameters


def community_detection():
    # Creating Pipeline
    pipeline_description = Pipeline(context='PRETRAINING')
    pipeline_description.add_input(name='inputs')
    # Step 0
    step_0 = PrimitiveStep(primitive_description=d3m.primitives.sri.graph.CommunityDetectionParser.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    add_hyperparameters(step_0, d3m.primitives.sri.graph.CommunityDetectionParser)
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1
    step_1 = PrimitiveStep(primitive_description=d3m.primitives.sri.psl.CommunityDetection.metadata.query())
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    add_hyperparameters(step_1, d3m.primitives.sri.psl.CommunityDetection)
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    pipeline_description.add_output(name='Result', data_reference='steps.1.produce')

    return pipeline_description