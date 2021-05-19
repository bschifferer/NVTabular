import glob
import tensorflow as tf
import nvtabular as nvt
import os
import numpy as np
import cudf
# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ['TF_MEMORY_ALLOCATION'] = "0.6" # fraction of free memory
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.framework_utils.tensorflow import layers
from nvtabular.inference.triton import export_tensorflow_ensemble

import tritonhttpclient
import tritonclient.http as httpclient
import nvtabular.inference.triton as nvt_triton 

import tritonclient.grpc as grpcclient

from tritonclient.utils import *

CATEGORICAL_COLUMNS = ['brand', 'productID', 'userID', 'cat_0', 'cat_1', 'cat_2', 'cat_3',
                       'unixReviewTime_toDatetime_day', 'unixReviewTime_toDatetime_month',
                       'unixReviewTime_toDatetime_year']
NUMERIC_COLUMNS = ['brand_c_count', 'productID_c_count', 'userID_c_count', 'cat_0_c_count',
                   'cat_1_c_count', 'cat_2_c_count', 'cat_3_c_count', 'price',
                  'salesRank_Electronics', 'salesRank_CellPhones', 'salesRank_Camera', 'salesRank_Computers',
                   'TE_brand_label', 'TE_productID_label',
                   'TE_userID_label', 'TE_cat_0_label',
                   'TE_cat_1_label', 'TE_cat_2_label',
                   'TE_cat_3_label',
                   'TE_unixReviewTime_toDatetime_month_label',
                   'TE_unixReviewTime_toDatetime_year_label']
LABEL_COLUMNS = ['label']
BATCH_SIZE = 1024
K = 2
BUFFER_SIZE = 0.06

def train_tensorflow():
    TRAIN_PATHS = sorted(glob.glob('./train_out/*.parquet'))
    VALID_PATHS = sorted(glob.glob('./valid_out/*.parquet'))
    train_dataset_tf = KerasSequenceLoader(
        TRAIN_PATHS, # you could also use a glob pattern
        batch_size=BATCH_SIZE,
        label_names=LABEL_COLUMNS,
        cat_names=CATEGORICAL_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine='parquet',
        shuffle=True,
        buffer_size=BUFFER_SIZE,
        parts_per_chunk=K
    )
    valid_dataset_tf = KerasSequenceLoader(
        VALID_PATHS, # you could also use a glob pattern
        batch_size=BATCH_SIZE,
        label_names=LABEL_COLUMNS,
        cat_names = CATEGORICAL_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine='parquet',
        shuffle=False,
        buffer_size=BUFFER_SIZE,
        parts_per_chunk=K
    )
    inputs = {}
    emb_layers = []
    num_layers = []
    for col in CATEGORICAL_COLUMNS:
        inputs[col] =  tf.keras.Input(
            name=col,
            dtype=tf.int32,
            shape=(1,)
        )
    for col in NUMERIC_COLUMNS:
        inputs[col] =  tf.keras.Input(
            name=col,
            dtype=tf.float32,
            shape=(1,)
        )
    workflow = nvt.Workflow.load('./nvtworkflow')
    embeddings = nvt.ops.get_embedding_sizes(workflow)
    for col in CATEGORICAL_COLUMNS:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, 
                    embeddings[col][0]                    # Input dimension (vocab size)
                ), embeddings[col][1]                     # Embedding output dimension
            )
        )
    for col in NUMERIC_COLUMNS:
        num_layers.append(
            tf.feature_column.numeric_column(col)
        )
    emb_layer = layers.DenseFeatures(emb_layers+num_layers)
    x_concat_output = emb_layer(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x_concat_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile('sgd', 'binary_crossentropy')
    validation_callback = KerasSequenceValidater(valid_dataset_tf)
    history = model.fit(train_dataset_tf, epochs=1, callbacks=[validation_callback])
    os.system('rm -rf tensorflow.savedmodel')
    model.save('./tensorflow.savedmodel')

def deploy_tensorflow():
    workflow = nvt.Workflow.load('./nvtworkflow')
    model = tf.keras.models.load_model('./tensorflow.savedmodel')
    for col in CATEGORICAL_COLUMNS:
        workflow.output_dtypes[col] = "int32"
    export_tensorflow_ensemble(model, workflow, "amazonreview", "/models/", LABEL_COLUMNS)
    triton_client = tritonhttpclient.InferenceServerClient(url="triton:8000", verbose=True)
    triton_client.load_model(model_name="amazonreview")

def test_inference():
    batch = cudf.read_parquet("./data/valid/*.parquet", num_rows=3)
    inputs = nvt_triton.convert_df_to_triton_input(['userID', 'productID', 'brand', 'price', 
                                                    'unixReviewTime', 'cat_0', 'cat_1', 'cat_2', 'cat_3'], 
                                                   batch, grpcclient.InferInput)
    outputs = [grpcclient.InferRequestedOutput("dense_3")]
    with grpcclient.InferenceServerClient("triton:8001") as client:
        response = client.infer("amazonreview", inputs, request_id="1", outputs=outputs)
    print("predicted sigmoid result:\n", response.as_numpy("dense_3"), "\n")

if __name__ == "__main__":
    train_tensorflow()
    deploy_tensorflow()
    test_inference()