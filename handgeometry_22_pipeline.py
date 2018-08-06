from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PipelineContext
import d3m
import d3m.primitives
import d3m.primitives.datasets
import d3m.primitives.dsbox
import d3m.primitive_interfaces
from modules.pipeline_generator.utils.hyperparams import add_hyperparameters


def handgeometry_dataset_pipline() -> Pipeline:
    # Creating Pipeline
    pipeline_description = Pipeline(context='PRETRAINING')
    pipeline_description.add_input(name='inputs')

    # Step 1: Denormalize
    step_0 = PrimitiveStep(primitive_description=d3m.primitives.dsbox.Denormalize.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: DatasetToDataFrame
    step_1 = PrimitiveStep(primitive_description=d3m.primitives.datasets.DatasetToDataFrame.metadata.query())
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    add_hyperparameters(step_1, d3m.primitives.data.DataFrameToList)
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 3:
    step_2 = PrimitiveStep(primitive_description=d3m.primitives.data.ExtractColumnsBySemanticTypes.metadata.query())
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                              data=["https://metadata.datadrivendiscovery.org/types/Target", "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"])
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)
    # Step 4:
    step_3 = PrimitiveStep(primitive_description=d3m.primitives.dsbox.DataFrameToTensor.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)
    # Step 5
    step_4 = PrimitiveStep(primitive_description=d3m.primitives.dsbox.Vgg16ImageFeature.metadata.query())
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)
    # Step 6
    step_5 = PrimitiveStep(primitive_description=d3m.primitives.sklearn_wrap.SKPCA.metadata.query())
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)
    # Step 7
    step_6 = PrimitiveStep(primitive_description=d3m.primitives.sklearn_wrap.SKRandomForestRegressor.metadata.query())
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)
    # Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')
    return pipeline_description