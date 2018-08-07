from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PipelineContext
import d3m
import d3m.primitives
import d3m.primitives.datasets
import d3m.primitives.sri.psl
import d3m.primitives.sri
import d3m.primitives.sri.graph
from modules.pipeline_generator.utils.hyperparams import add_hyperparameters

def collaborative_filtering_link_prediction():
    # Creating Pipeline
    pipeline_description = Pipeline(context='PRETRAINING')
    pipeline_description.add_input(name='inputs')

    # Step 0: GraphMatchingParser
    step_0 = PrimitiveStep(primitive_description=d3m.primitives.sri.graph.CollaborativeFilteringParser.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: Apply GraphTransformer
    step_1 = PrimitiveStep(primitive_description=d3m.primitives.sri.graph.GraphTransformer.metadata.query())
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 2: Apply LinkPrediction
    step_2 = PrimitiveStep(primitive_description=d3m.primitives.sri.psl.LinkPrediction.metadata.query())
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_hyperparameter(name='prediction_column', argument_type=ArgumentType.VALUE,
                              data="rating")
    step_2.add_hyperparameter(name='truth_threshold', argument_type=ArgumentType.VALUE,
                              data=1e-07)
    step_2.add_hyperparameter(name="jvm_memory", argument_type=ArgumentType.VALUE, data=0.5)
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    # Step 3: ConstructPredictions
    step_3 = PrimitiveStep(primitive_description=d3m.primitives.data.ConstructPredictions.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_hyperparameter(name='use_columns', argument_type=ArgumentType.VALUE, data=[0, 1])
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)

    # Step 4: RemoveColumns
    step_4 = PrimitiveStep(primitive_description=d3m.primitives.data.RemoveColumns.metadata.query())
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_hyperparameter(name='columns', argument_type=ArgumentType.VALUE, data=[0])
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    pipeline_description.add_output(name='Result', data_reference='steps.4.produce')

    return pipeline_description
