from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PipelineContext
import d3m
import d3m.primitives
import d3m.primitives.datasets
import d3m.primitives.sri.psl
import d3m.primitives.sri
import d3m.primitives.sri.graph

from modules.pipeline_generator.utils import custom_resolver
from modules.pipeline_generator.utils import pipelines


def community_detection(resolver=None):
    if resolver is None:
        resolver = custom_resolver.BlackListResolver()

    # Creating Pipeline
    pipeline_description = Pipeline(context=PipelineContext.TESTING)
    pipeline_description.add_input(name='inputs')
    start_step = "inputs.0"

    # Step 0
    step_0 = PrimitiveStep(primitive_description=d3m.primitives.sri.graph.CommunityDetectionParser.metadata.query(), resolver=resolver)
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=start_step)
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1
    step_1 = PrimitiveStep(primitive_description=d3m.primitives.sri.psl.CommunityDetection.metadata.query(), resolver=resolver)
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_hyperparameter(name='jvm_memory', argument_type=ArgumentType.VALUE, data=0.5)
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 2: RemoveColumns
    step_2 = PrimitiveStep(primitive_description=d3m.primitives.data.RemoveColumns.metadata.query(), resolver=resolver)
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_hyperparameter(name='columns', argument_type=ArgumentType.VALUE, data=[0])
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    pipeline_description.add_output(name='Result', data_reference='steps.2.produce')

    last_step = len(pipeline_description.steps) - 1
    attributes = pipelines.int_to_step(last_step - 1)
    targets = pipelines.int_to_step(last_step)

    return pipeline_description

