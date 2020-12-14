#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import cudf
import cupy

from . import categorify as nvt_cat
from .stat_operator import StatOperator


class TargetEncoding(StatOperator):
    """
    Target encoding is a common feature-engineering technique for
    categorical columns in tabular datasets. For each categorical group,
    the mean of a continuous target column is calculated, and the
    group-specific mean of each row is used to create a new feature (column).
    To prevent overfitting, the following additional logic is applied:

        1. Cross Validation: To prevent overfitting in training data,
        a cross-validation strategy is used - The data is split into
        k random "folds", and the mean values within the i-th fold are
        calculated with data from all other folds. The cross-validation
        strategy is only employed when the dataset is used to update
        recorded statistics. For transformation-only workflow execution,
        global-mean statistics are used instead.

        2. Smoothing: To prevent overfitting for low cardinality categories,
        the means are smoothed with the overall mean of the target variable.

    Function:
    TE = ((mean_cat*count_cat)+(mean_global*p_smooth)) / (count_cat+p_smooth)

    count_cat := count of the categorical value
    mean_cat := mean of target value for the categorical value
    mean_global := mean of the target value in the dataset
    p_smooth := smoothing factor

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add TE op to the workflow
        proc.add_feature(
            TargetEncoding(
            cat_groups = ['cat1', 'cat2', ['cat2','cat3']],
            target = LABEL_COLUMNS,
            kfold = 5,
            p_smooth = 20)
        )

    Parameters
    -----------
    target : str
        Continuous target column to use for the encoding of cat_groups.
        The same continuous target will be used for all `cat_groups`.
    target_mean : float
        Global mean of the target column to use for encoding.
        Supplying this value up-front will improve performance.
    kfold : int, default 3
        Number of cross-validation folds to use while gathering
        statistics (during `GroupbyStatistics`).
    fold_seed : int, default 42
        Random seed to use for cupy-based fold assignment.
    drop_folds : bool, default True
        Whether to drop the "__fold__" column created by the
        `GroupbyStatistics` dependency (after the transformation).
    p_smooth : int, default 20
        Smoothing factor.
    out_col : str or list of str, default is problem-specific
        Name of output target-encoding column. If `cat_groups` includes
        multiple elements, this should be a list of the same length (and
        elements must be unique).
    out_dtype : str, default is problem-specific
        dtype of output target-encoding columns.
    tree_width : dict or int, optional
        Passed to `GroupbyStatistics` dependency.
    out_path : str, optional
        Passed to `GroupbyStatistics` dependency.
    on_host : bool, default True
        Passed to `GroupbyStatistics` dependency.
    name_sep : str, default "_"
        String separator to use between concatenated column names
        for multi-column groups.
    """

    def __init__(
        self,
        target,
        target_mean=None,
        kfold=None,
        fold_seed=42,
        p_smooth=20,
        out_col=None,
        out_dtype=None,
        tree_width=None,
        cat_cache="host",
        out_path=None,
        on_host=True,
        name_sep="_",
        stat_name=None,
        drop_folds=True,
    ):
        super().__init__()

        self.target = [target] if isinstance(target, str) else target
        self.dependency = self.target

        if hasattr(self.target, "columns"):
            self.target = self.target.columns

        self.target_mean = target_mean
        self.kfold = kfold or 3
        self.fold_seed = fold_seed
        self.p_smooth = p_smooth
        self.out_col = [out_col] if isinstance(out_col, str) else out_col
        self.out_dtype = out_dtype
        self.tree_width = tree_width
        self.out_path = out_path
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.name_sep = name_sep
        self.drop_folds = drop_folds
        self.stat_name = stat_name or "te_stats"

    # TODO: fit/fit_finalize methods

    def dependencies(self):
        return self.dependency

    def output_column_names(self, columns):
        ret = []
        for cat in columns:
            cat = [cat] if isinstance(cat, str) else cat
            ret.extend(self._make_te_name(cat))
        return ret

    def _make_te_name(self, cat_group):
        tag = nvt_cat._make_name(*cat_group, sep=self.name_sep)
        return [f"TE_{tag}_{x}" for x in self.target]

    def _op_group_logic(self, cat_group, gdf, stats_context, y_mean, fit_folds, group_ind):

        # Define name of new TE column
        if isinstance(self.out_col, list):
            if group_ind >= len(self.out_col):
                raise ValueError("out_col and cat_groups are different sizes.")
            out_col = self.out_col[group_ind]
            out_col = [out_col] if isinstance(out_col, str) else out_col
            # ToDo Test
            if len(out_col) != len(self.target):
                raise ValueError("out_col and target are different sizes.")
        else:
            out_col = self._make_te_name(cat_group)

        # Initialize new data
        new_gdf = cudf.DataFrame()
        tmp = "__tmp__"

        if fit_folds:
            # Groupby Aggregation for each fold
            cols = ["__fold__"] + cat_group
            storage_name_folds = nvt_cat._make_name(*cols, sep=self.name_sep)
            path_folds = stats_context[self.stat_name][storage_name_folds]
            agg_each_fold = nvt_cat._read_groupby_stat_df(
                path_folds, storage_name_folds, self.cat_cache
            )
            agg_each_fold.columns = cols + ["count_y"] + [x + "_sum_y" for x in self.target]
        else:
            cols = cat_group

        # Groupby Aggregation for all data
        storage_name_all = nvt_cat._make_name(*cat_group, sep=self.name_sep)
        path_all = stats_context[self.stat_name][storage_name_all]
        agg_all = nvt_cat._read_groupby_stat_df(path_all, storage_name_all, self.cat_cache)
        agg_all.columns = cat_group + ["count_y_all"] + [x + "_sum_y_all" for x in self.target]

        if fit_folds:
            agg_each_fold = agg_each_fold.merge(agg_all, on=cat_group, how="left")
            agg_each_fold["count_y_all"] = agg_each_fold["count_y_all"] - agg_each_fold["count_y"]
            for i, x in enumerate(self.target):
                agg_each_fold[x + "_sum_y_all"] = (
                    agg_each_fold[x + "_sum_y_all"] - agg_each_fold[x + "_sum_y"]
                )
                agg_each_fold[out_col[i]] = (
                    agg_each_fold[x + "_sum_y_all"] + self.p_smooth * y_mean[x]
                ) / (agg_each_fold["count_y_all"] + self.p_smooth)

            agg_each_fold = agg_each_fold.drop(
                ["count_y_all", "count_y"]
                + [x + "_sum_y" for x in self.target]
                + [x + "_sum_y_all" for x in self.target],
                axis=1,
            )
            tran_gdf = gdf[cols + [tmp]].merge(agg_each_fold, on=cols, how="left")
            del agg_each_fold
        else:
            for i, x in enumerate(self.target):
                agg_all[out_col[i]] = (agg_all[x + "_sum_y_all"] + self.p_smooth * y_mean[x]) / (
                    agg_all["count_y_all"] + self.p_smooth
                )
            agg_all = agg_all.drop(
                ["count_y_all"] + [x + "_sum_y_all" for x in self.target], axis=1
            )
            tran_gdf = gdf[cols + [tmp]].merge(agg_all, on=cols, how="left")
            del agg_all

        # TODO: There is no need to perform the `agg_each_fold.merge(agg_all, ...)` merge
        #     for every partition.  We can/should cache the result for better performance.

        for i, x in enumerate(self.target):
            tran_gdf[out_col[i]] = tran_gdf[out_col[i]].fillna(y_mean[x])
        if self.out_dtype is not None:
            tran_gdf[out_col] = tran_gdf[out_col].astype(self.out_dtype)

        tran_gdf = tran_gdf.sort_values(tmp, ignore_index=True)
        tran_gdf.drop(columns=cols + [tmp], inplace=True)
        new_cols = [c for c in tran_gdf.columns if c not in new_gdf.columns]
        new_gdf[new_cols] = tran_gdf[new_cols]

        # Make sure we are preserving the index of gdf
        new_gdf.index = gdf.index
        return new_gdf

    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):

        # Add temporary column for sorting
        tmp = "__tmp__"
        gdf[tmp] = cupy.arange(len(gdf), dtype="int32")

        # Only perform "fit" if fold column is present
        fit_folds = "__fold__" in gdf.columns

        # Need mean of contiuous target column
        y_mean = self.target_mean or stats_context["means"]

        # Loop over categorical-column groups and apply logic
        new_gdf = None
        for ind, cat_group in enumerate(self.cat_groups):
            if new_gdf is None:
                new_gdf = self._op_group_logic(
                    cat_group, gdf, stats_context, y_mean, fit_folds, ind
                )
            else:
                _df = self._op_group_logic(cat_group, gdf, stats_context, y_mean, fit_folds, ind)
                new_gdf = cudf.concat([new_gdf, _df], axis=1)

        # Drop temporary columns
        gdf.drop(
            columns=[tmp, "__fold__"] if fit_folds and self.drop_folds else [tmp], inplace=True
        )
        return new_gdf
