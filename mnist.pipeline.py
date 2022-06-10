import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    packages_to_install=["tensorflow", "numpy"]
)
def load_data(output_dataset: Output[Dataset]):
    print("data loading...")
    import tensorflow as tf
    import numpy as np

    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    with open(output_dataset.path, "wb") as f:
        np.savez(f, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    print(f"Saved raw data on : {output_dataset.path}")

@component(
    packages_to_install=["numpy"]
)
def preprocessing(input_dataset: Input[Dataset], output_dataset: Output[Dataset]):
    print("Preprocessing...")
    import numpy as np

    with open(input_dataset.path, "rb") as f:
        mnist = np.load(f)
        train_x, train_y = mnist["train_x"], mnist["train_y"]
        test_x, test_y = mnist["test_x"], mnist["test_y"]

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    with open(output_dataset.path, "wb") as f:
        np.savez(f, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    print(f"Saved preproceesed data on : {output_dataset.path}")

@component(
    packages_to_install=["tensorflow", "numpy"]
)
def train(
    dataset: Input[Dataset], output_model: Output[Model], metrics: Output[Metrics]
):
    print("training...")
    import tensorflow as tf
    import numpy as np

    with open(dataset.path, "rb") as f:
        mnist = np.load(f)
        train_x, train_y = mnist["train_x"], mnist["train_y"]
        test_x, test_y = mnist["test_x"], mnist["test_y"]
    print(f"train x shape: {train_x.shape}")
    print(f"train y shape: {train_y.shape}")
    print(f"test x shape: {test_x.shape}")
    print(f"test y shape: {test_y.shape}")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )
    model.fit(train_x, train_y)
    loss, acc = model.evaluate(test_x, test_y)

    metrics.log_metric("accuracy", (acc * 100.0))
    metrics.log_metric("loss", loss)
    metrics.log_metric("framework", "Tensorflow")
    metrics.log_metric("Model", "LinearModel")
    metrics.log_metric("dataset_size", len(train_x))

    model.save(output_model.path)

@dsl.pipeline(
    name="mnist-pipeline", description="An example pipeline that mnist training."
)
def mnist_pipeline():
    load_data_task = load_data()
    print("outputs: ", load_data_task.output)
    preprocessing_task = preprocessing(load_data_task.outputs["output_dataset"])
    train_task = train(preprocessing_task.outputs["output_dataset"])

client = kfp.Client()
client.create_run_from_pipeline_func(
    mnist_pipeline, arguments={}, mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
)