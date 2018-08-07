from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PipelineContext
import d3m
import d3m.primitives
import d3m.primitives.datasets
import d3m.primitives.dsbox
import d3m.primitive_interfaces
from modules.pipeline_generator.utils.hyperparams import add_hyperparameters
from modules.pipeline_generator.utils import custom_resolver
from modules.pipeline_generator.utils import pipelines


def image_regress_pipeline(resolver=None) -> Pipeline:

    if resolver is None:
        resolver = custom_resolver.BlackListResolver()
    # Creating Pipeline
    pipeline_description = Pipeline(context='PRETRAINING')
    pipeline_description.add_input(name='inputs')

    start_step = "inputs.0"

    # Step 1: Denormalize
    step_0 = PrimitiveStep(primitive_description=d3m.primitives.dsbox.Denormalize.metadata.query(), resolver=resolver)
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=start_step)
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: DatasetToDataFrame
    step_1 = PrimitiveStep(primitive_description=d3m.primitives.datasets.DatasetToDataFrame.metadata.query(), resolver=resolver)
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    add_hyperparameters(step_1, d3m.primitives.data.DataFrameToList)
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 3: Extract Target Column
    step_2 = PrimitiveStep(primitive_description=d3m.primitives.data.ExtractColumnsBySemanticTypes.metadata.query(), resolver=resolver)
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                              data=["https://metadata.datadrivendiscovery.org/types/Target", "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"])
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    # Step 4: Dataframe to tensor
    step_3 = PrimitiveStep(primitive_description=d3m.primitives.dsbox.DataFrameToTensor.metadata.query(), resolver=resolver)
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)

    # Step 5: Vgg16 Feature Extractor
    step_4 = PrimitiveStep(primitive_description=d3m.primitives.dsbox.Vgg16ImageFeature.metadata.query(), resolver=resolver)
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # Step 6: Apply PCA to feature generated
    step_5 = PrimitiveStep(primitive_description=d3m.primitives.sklearn_wrap.SKPCA.metadata.query(), resolver=resolver)
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Step 7: Apply Random Forest Regressor
    step_6 = PrimitiveStep(primitive_description=d3m.primitives.sklearn_wrap.SKRandomForestRegressor.metadata.query(), resolver=resolver)
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)

    # Output Generated
    pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

    last_step = len(pipeline_description.steps) - 1
    attributes = pipelines.int_to_step(last_step - 1)
    targets = pipelines.int_to_step(last_step)

    return pipeline_description
