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
from nvtx import annotate

from .operator import Operator
from .stat_operator import StatOperator


class FillMissing(Operator):
    """
    This operation replaces missing values with a constant pre-defined value

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add FillMissing to the workflow for continuous columns and specify the fill value
        # Default is 0
        proc.add_cont_feature(nvt.ops.FillMissing(fill_val=100))

    Parameters
    -----------
    fill_val : float, default 0
        The constant value to replace missing values with.
    """

    def __init__(self, fill_val=0):
        super().__init__()
        self.fill_val = fill_val

    @annotate("FillMissing_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns, gdf: cudf.DataFrame):
        return gdf[columns].fillna(self.fill_val)


class FillMedian(StatOperator):
    """
    This operation replaces missing values with the median value for the column.

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add FillMedian to the workflow for continuous columns
        proc.add_cont_feature(nvt.ops.FillMedian())
    """

    def __init__(self):
        super().__init__()
        self.medians = {}

    @annotate("FillMedian_transform", color="darkgreen", domain="nvt_python")
    def transform(self, columns, gdf: cudf.DataFrame):
        if not self.medians:
            raise RuntimeError("need to call 'fit' before running transform")

        for col in columns:
            gdf[col] = gdf[col].fillna(self.medians[col])
        return gdf

    @annotate("FillMedian_fit", color="green", domain="nvt_python")
    def fit(self, columns, ddf):
        # TODO: Use `method="tidigest"` when crick supports device
        dask_stats = ddf[columns].quantile(q=0.5, method="dask")
        return dask_stats

    @annotate("FillMedian_finalize", color="green", domain="nvt_python")
    def fit_finalize(self, dask_stats):
        for col in dask_stats.index.values_host:
            self.medians[col] = float(dask_stats[col])
