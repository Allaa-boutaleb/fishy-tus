# table-union-benchmarks/profiler

These two scripts can be used to obtain basic benchmark statistics as well as examine the schema and value overlap of ground truth unionable pairs in each benchmark.

## Example usage:

```
python profiler/profile_benchmark.py data/santos profiles
python profiler/benchmark_overlap.py data/santos profiles
```

`profile_benchmark.py` will create a json file (`santos_overall.json`) in a directory called `profiles/`, which shows basic stats of the benchmark as described in Table 1 of the paper. 
`benchmark_overlap.py` will create two json files: (`santos_overlap_summary.json`) and (`santos_overlap_detailed.json`). The first one contains overall overlap statistics whereas the second one measures the overlap for each query table and its ground truth unionable pairs as well.



