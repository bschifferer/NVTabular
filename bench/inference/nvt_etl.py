import glob
import os

import cudf

import numpy as np
import nvtabular as nvt

def nvt_etl():
    os.system('rm -rf categories')
    ### Introduction NVTabular
    cat_features = (
        (['brand'] >> nvt.ops.FillMissing()) + 
        ['productID', 'userID', 'cat_0', 'cat_1', 'cat_2', 'cat_3'] >> nvt.ops.Categorify()
    )
    ### 1. FillMedian and Normalize
    cont_features = ['price'] >> nvt.ops.FillMedian() >> nvt.ops.Normalize()
    ### 2. LogOp
    count_encode = (
        cat_features >> 
        nvt.ops.Rename(postfix="_c") >> # We rename the columns to avoid naming conflict in calc. statistics
        nvt.ops.FillMissing() >>
        nvt.ops.JoinGroupby(cont_cols=['label'], stats=["count"]) >>
        nvt.ops.LogOp() >>
        nvt.ops.Normalize() >> 
        nvt.ops.FillMissing()
    )
    ### 3. LambdaOp
    unixReviewTime_toDatetime = (
        ['unixReviewTime'] >> 
        nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
        nvt.ops.Rename(postfix="_toDatetime")
    )
    label_tobinary = (
        ['label'] >> 
        nvt.ops.LambdaOp(lambda col: (col>3.0).astype('int8'))
    )
    unixReviewTime_toDatetime_day = (
        unixReviewTime_toDatetime >> 
        nvt.ops.LambdaOp(lambda col: col.dt.day) >> 
        nvt.ops.Rename(postfix="_day")
    )
    unixReviewTime_toDatetime_month = (
        unixReviewTime_toDatetime >> 
        nvt.ops.LambdaOp(lambda col: col.dt.month) >> 
        nvt.ops.Rename(postfix="_month")
    )
    unixReviewTime_toDatetime_year = (
        unixReviewTime_toDatetime >> 
        nvt.ops.LambdaOp(lambda col: col.dt.year) >> 
        nvt.ops.Rename(postfix="_year")
    )
    time_features = (
        unixReviewTime_toDatetime_day + 
        unixReviewTime_toDatetime_month + 
        unixReviewTime_toDatetime_year
    ) >> nvt.ops.Categorify()
    ### 4. JoinExternal
    df_ext = cudf.read_parquet('./data/salesrank.parquet')
    joined = nvt.ColumnGroup(['productID']) >> nvt.ops.JoinExternal(
        df_ext,
        on =['productID'],
        how='left'
    )
    joined_na = joined - ['productID', 
                          'salesRank_Electronics', 
                          'salesRank_Camera', 
                          'salesRank_Computers', 
                          'salesRank_CellPhones']
    joined_norm = joined - ['productID', 
                            'salesRank_Electronics_NA', 
                            'salesRank_Camera_NA', 
                            'salesRank_Computers_NA', 
                            'salesRank_CellPhones_NA']
    joined_norm = joined_norm >> nvt.ops.Normalize()

    ### 5. & 6. TargetEncoding with NormalizeMinMax
    te_cats = [
        ['userID', 'brand'], ['userID', 'cat_2'], ['cat_0', 'cat_1', 'cat_2'],['userID', 'brand', 'cat_2']
    ]
    te_features = cat_features + unixReviewTime_toDatetime_month + unixReviewTime_toDatetime_year >> nvt.ops.TargetEncoding(
        label_tobinary, 
        kfold=1, 
        p_smooth=20
    ) >> nvt.ops.NormalizeMinMax()
    
    features = (
        cat_features + 
        count_encode + 
        cont_features + 
        time_features +
        joined_norm +
        te_features +
        label_tobinary
    )
    
    # 2. Initialize the NVTabular workflow
    workflow = nvt.Workflow(features)

    # 3. Initialize the NVTabular dataset
    TRAIN_PATHS = sorted(glob.glob(os.path.join('./data/train/', '*.parquet')))
    VALID_PATHS = sorted(glob.glob(os.path.join('./data/valid/', '*.parquet')))
    train_dataset = nvt.Dataset(TRAIN_PATHS, engine='parquet')
    valid_dataset = nvt.Dataset(VALID_PATHS, engine='parquet')
    
    # 4. Fit the train dataset to collect statistics/information
    workflow.fit(train_dataset)

    # 5. Transform the train and valid dataset
    os.system('rm -rf ./train_out/')
    os.system('rm -rf ./valid_out/')
    workflow.transform(train_dataset).to_parquet(output_path='./train_out/', 
                                                 shuffle=nvt.io.Shuffle.PER_PARTITION)
    workflow.transform(valid_dataset).to_parquet(output_path='./valid_out/')
    os.system('rm -rf ./nvtworkflow/')
    workflow.save('./nvtworkflow')

if __name__ == "__main__":
    nvt_etl()