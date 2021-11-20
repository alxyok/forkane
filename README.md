# forkane

An toy experiment with Netflix's Metaflow framework.

Building graphs from RAISE's Open Source data. Each data file contains three H5 datasets of grid shape `(65, 33, 33)`. For the toy example, to produce the dataset, we need to:

1. Convert the H5 file to a Numpy file.
2. Build the graph with a PyTorch-Geometric Data structure.
3. Serialize the graph to a PyTorch object.

To download the data from RAISE's resources, run the `download-data.sh`.

`run.sh` to run the entire pipeline.