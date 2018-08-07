from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType
import d3m
import d3m.primitives
import d3m.primitives.datasets
import d3m.primitives.bbn
import d3m.primitives.sklearn_wrap
import d3m.primitives.bbn.time_series
from modules.pipeline_generator.utils.hyperparams import add_hyperparameters


def audio_pipeline():
    pipeline_description = Pipeline(context='TESTING')
    pipeline_description.add_input(name='inputs')
     # Step 0
    step_0 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.TargetsReader.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    add_hyperparameters(step_0, d3m.primitives.bbn.time_series.TargetsReader)
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)
    # Step 1
    step_1 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.AudioReader.metadata.query())
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    add_hyperparameters(step_1, d3m.primitives.bbn.time_series.AudioReader)
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)
    # Step 2
    step_2 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.ChannelAverager.metadata.query())
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    add_hyperparameters(step_2, d3m.primitives.bbn.time_series.ChannelAverager)
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)
    # Step 3
    step_3 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.SignalDither.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    add_hyperparameters(step_3, d3m.primitives.bbn.time_series.SignalDither)
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)
    # Step 4
    step_4 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.SignalFramer.metadata.query())
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    add_hyperparameters(step_4, d3m.primitives.bbn.time_series.SignalFramer)
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)
    # Step 5
    step_5 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.SignalMFCC.metadata.query())
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    add_hyperparameters(step_5, d3m.primitives.bbn.time_series.SignalMFCC)
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)
    # Step 6
    step_6 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.UniformSegmentation.metadata.query())
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    add_hyperparameters(step_6, d3m.primitives.bbn.time_series.UniformSegmentation)
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)
    # Step 7
    step_7 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.SegmentCurveFitter.metadata.query())
    step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
    add_hyperparameters(step_7, d3m.primitives.bbn.time_series.SegmentCurveFitter)
    step_7.add_output('produce')
    pipeline_description.add_step(step_7)
    # Step 8
    step_8 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.ClusterCurveFittingKMeans.metadata.query())
    step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.7.produce')
    add_hyperparameters(step_8, d3m.primitives.bbn.time_series.ClusterCurveFittingKMeans)
    step_8.add_output('produce')
    pipeline_description.add_step(step_8)
    # Step 9
    step_9 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.SignalFramer.metadata.query())
    step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.8.produce')
    add_hyperparameters(step_9, d3m.primitives.bbn.time_series.SignalFramer)
    step_9.add_output('produce')
    pipeline_description.add_step(step_9)
    # Step 10
    step_10 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.SequenceToBagOfTokens.metadata.query())
    step_10.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.9.produce')
    add_hyperparameters(step_10, d3m.primitives.bbn.time_series.SequenceToBagOfTokens)
    step_10.add_output('produce')
    pipeline_description.add_step(step_10)
    # Step 11
    step_11 = PrimitiveStep(primitive_description=d3m.primitives.bbn.time_series.BBNTfidfTransformer.metadata.query())
    step_11.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.10.produce')
    add_hyperparameters(step_11, d3m.primitives.bbn.time_series.BBNTfidfTransformer)
    step_11.add_output('produce')
    pipeline_description.add_step(step_11)
    # Step 12
    step_12 = PrimitiveStep(primitive_description=d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier.metadata.query())
    step_12.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.11.produce')
    step_12.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    add_hyperparameters(step_12, d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier)
    step_12.add_output('produce')
    pipeline_description.add_step(step_12)

    # final output
    pipeline_description.add_output(name='output predictions', data_reference='steps.12.produce')
    return pipeline_description