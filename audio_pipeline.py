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
    step_0 = PrimitiveStep(primitive_description=primitives.bbn.time_series.AudioReader.metadata.query())
    step_0.add_argument(name='inputs', argument_type='CONTAINER', data_reference='inputs.0')
    add_hyperparameters(step_0, primitives.bbn.time_series.AudioReader)
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)
    # Step 1
    step_1 = PrimitiveStep(primitive_description=primitives.bbn.time_series.ChannelAverager.metadata.query())
    step_1.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.0.produce')
    add_hyperparameters(step_1, primitives.bbn.time_series.ChannelAverager)
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)
    # Step 2
    step_2 = PrimitiveStep(primitive_description=primitives.bbn.time_series.SignalDither.metadata.query())
    step_2.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.1.produce')
    add_hyperparameters(step_2, primitives.bbn.time_series.SignalDither)
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)
    # Step 3
    step_3 = PrimitiveStep(primitive_description=primitives.bbn.time_series.SignalFramer.metadata.query())
    step_3.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.2.produce')
    add_hyperparameters(step_3, primitives.bbn.time_series.SignalFramer)
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)
    # Step 4
    step_4 = PrimitiveStep(primitive_description=primitives.bbn.time_series.SignalMFCC.metadata.query())
    step_4.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.3.produce')
    add_hyperparameters(step_4, primitives.bbn.time_series.SignalMFCC)
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)
    # Step 5
    step_5 = PrimitiveStep(primitive_description=primitives.bbn.time_series.UniformSegmentation.metadata.query())
    step_5.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.4.produce')
    add_hyperparameters(step_5, primitives.bbn.time_series.UniformSegmentation)
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)
    # Step 6
    step_6 = PrimitiveStep(primitive_description=primitives.bbn.time_series.SegmentCurveFitter.metadata.query())
    step_6.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.5.produce')
    add_hyperparameters(step_6, primitives.bbn.time_series.SegmentCurveFitter)
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)
    # Step 7
    step_7 = PrimitiveStep(primitive_description=primitives.bbn.time_series.ClusterCurveFittingKMeans.metadata.query())
    step_7.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.6.produce')
    add_hyperparameters(step_7, primitives.bbn.time_series.ClusterCurveFittingKMeans)
    step_7.add_output('produce')
    pipeline_description.add_step(step_7)
    # Step 8
    step_8 = PrimitiveStep(primitive_description=primitives.bbn.time_series.SignalFramer.metadata.query())
    step_8.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.7.produce')
    add_hyperparameters(step_8, primitives.bbn.time_series.SignalFramer)
    step_8.add_output('produce')
    pipeline_description.add_step(step_8)
    # Step 9
    step_9 = PrimitiveStep(primitive_description=primitives.bbn.time_series.SequenceToBagOfTokens.metadata.query())
    step_9.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.8.produce')
    add_hyperparameters(step_9, primitives.bbn.time_series.SequenceToBagOfTokens)
    step_9.add_output('produce')
    pipeline_description.add_step(step_9)
    # Step 10
    step_10 = PrimitiveStep(primitive_description=primitives.bbn.time_series.BBNTfidfTransformer.metadata.query())
    step_10.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.9.produce')
    add_hyperparameters(step_10, primitives.bbn.time_series.BBNTfidfTransformer)
    step_10.add_output('produce')
    pipeline_description.add_step(step_10)
    # Step 11
    step_11 = PrimitiveStep(primitive_description=primitives.datasets.Denormalize.metadata.query())
    step_11.add_argument(name='inputs', argument_type='CONTAINER', data_reference='inputs.0')
    add_hyperparameters(step_11, primitives.datasets.Denormalize)
    step_11.add_output('produce')
    pipeline_description.add_step(step_11)
    # Step 12
    step_12 = PrimitiveStep(primitive_description=primitives.datasets.DatasetToDataFrame.metadata.query())
    step_12.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.11.produce')
    add_hyperparameters(step_12, primitives.datasets.DatasetToDataFrame)
    step_12.add_output('produce')
    pipeline_description.add_step(step_12)
    # Step 13
    step_13 = PrimitiveStep(primitive_description=primitives.data.ExtractTargets.metadata.query())
    step_13.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.12.produce')
    add_hyperparameters(step_13, primitives.data.ExtractTargets)
    step_13.add_output('produce')
    pipeline_description.add_step(step_13)
    # Step 14
    step_14 = PrimitiveStep(primitive_description=primitives.sklearn_wrap.SKLinearSVC.metadata.query())
    step_14.add_argument(name='inputs', argument_type='CONTAINER', data_reference='steps.10.produce')
    step_14.add_argument(name='outputs', argument_type='CONTAINER', data_reference='steps.13.produce')
    add_hyperparameters(step_14, primitives.sklearn_wrap.SKLinearSVC)
    step_14.add_output('produce')
    pipeline_description.add_step(step_14)

    # final output
    pipeline_description.add_output(name='output predictions', data_reference='steps.14.produce')

    return pipeline_description
