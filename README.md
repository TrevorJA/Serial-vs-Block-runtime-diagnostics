# Serial-vs-Block-runtime-diagnostics

## Summary
Contains scripts comparing two different implementations of many-scenario/realization evaluation for the Shallow Lake Problem (Carpenter et al., 1999; Quinn et al., 2017).

[A blog post describing both methods, and the details associated with the diagnostic test can be found here](https://waterprogramming.wordpress.com/2022/06/14/a-time-saving-tip-for-simulating-multiple-scenarios/). 

Additionally, (not described in the above blog post) memory profiling is able to be performed for both implementations, and runtime comparisons are performed.

## Content

### Models
[intertemporal_lake_model_serial_implementation.py](https://github.com/TrevorJA/Serial-vs-Block-runtime-diagnostics/blob/main/intertemporal_lake_model_series_implementation.py)
> Contains the lake model, for an intertemporal pollution policy (100 annual pollution release decisions for 100 years), implemented using the serial simulation method for many-scenario evaluation. A single scenario (unique realization of inflow stochasticity) is evaluated for the entire simulation period before moving onto the next scenario. 

[intertemporal_lake_model_block_implementation.py](https://github.com/TrevorJA/Serial-vs-Block-runtime-diagnostics/blob/main/intertemporal_lake_model_block_implementation.py)
> Contains the lake model, for an intertemporal pollution policy (100 annual pollution release decisions for 100 years), implemented using the block, or vector, simulation method for many-scenario evaluation. All scenarios (unique realizations of inflow stochasticity) are evaluated for a certain time step before moving onto the next time step.

### Diagnostics
[runtime_comparison.py](https://github.com/TrevorJA/Serial-vs-Block-runtime-diagnostics/blob/main/runtime_comparison.py)
> Measures the runtime for both the serial and block methods of simulating many scenarios/realizations. Timing tests are performed for a range of scenario ensemble sizes, from 25 to 1000 scenarios. The relative speed of the block method with respect to the serial implementation is plotted. The block method is shown to be significantly faster.

[memory_profiling_block.py](https://github.com/TrevorJA/Serial-vs-Block-runtime-diagnostics/blob/main/memory_profiling_block.py)
> Uses the [memory_profiler module](https://pypi.org/project/memory-profiler/) to profile the memory usage of the block implmentation. Run this script from a command line, by calling: _python -m memory_profiler memory_profiling_block.py_
> Plot the memory usage by calling, from the command line: _mprof run memory_profiling_block.py_ followed by _mprof plot_ to produce a plot of memory usage with respect to time. 

[memory_profiling_series.py](https://github.com/TrevorJA/Serial-vs-Block-runtime-diagnostics/blob/main/memory_profiling_series.py)
> Uses the [memory_profiler module](https://pypi.org/project/memory-profiler/) to profile the memory usage of the series implmentation.
> Use the command line code described above to execute the script. 

### Data
[optimized_intertemporal_pollution_policy_data.resultfile](https://github.com/TrevorJA/Serial-vs-Block-runtime-diagnostics/blob/main/optimized_intertemporal_pollution_policy_data.resultfile)
> Contains 56 Pareto-optimal intertemporal pollution policies, consisting of 100 annual pollution release values. Solutions were optimized following the methods described by [Quinn et al. (2017)](https://www.sciencedirect.com/science/article/pii/S1364815216302250). The last 4 columns of every row contain the objective scores. Only 1 solution was used from this set for demonstration purposes. 
