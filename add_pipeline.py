import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import component

@component
def add(a: float, b: float) -> float:
    """Calculates sum of two arguments"""
    return a + b

@dsl.pipeline(
    name="addition-pipeline",
    description="An example pipeline that performs addition calculations.",
    # pipeline_root='gs://my-pipeline-root/example-pipeline'
)
def add_pipeline(a: float = 1, b: float = 7):
    add_task = add(a, b)

client = kfp.Client()
client.create_run_from_pipeline_func(
    add_pipeline,
    arguments={"a": 7, "b": 8},
    mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
)