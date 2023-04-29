# qe-tools

A library for interfacing with the Quantum-Espresso output.  

# Example Utilities

## Structure band energy data across a uniform grid of k-space
QE is is great at computing and displaying band structure in the form of 1D traces, but it is harder to simulate a mesh of the BZ and keeping data structured.

We have written tools for generating (arbitrary sized) Monkhorst-Pack grids of kpoints and keeping the data structured in the dimensions of k-space. 
The data is structured via a subclass of the [WrightTools Data object](http://wright.tools/en/stable/data.html).

First set up a config file with lattice and grid parameters:
```toml
# config.toml
[lattice]
a1 = [1.0, 0.0, 0.0]
a2 = [-0.5, 0.866025, 0.0,]
a3 = [0.0, 0.0, 3.0,]

[grid]
ns = [100, 100, 1]

[options]
fermi = 1.0545  # eV
```

Run:
```python
>>> my_list = qe_tools.gen_klist(toml_path="config.toml", target="output.txt")
# use output.txt to as input to qe, then grab the qe output file e.g. "bands.sample.txt".
>>> data = qe_tools.as_structured(toml_path="config.toml", band_path="bands.sample.txt")  # data is a `MeshData` object
>>> print(data.shape)
(100,100,1)
```

## Generate p-matrix data structured across k-space
_coming soon_

## The MeshData object
Structured data has all the friendly utilities of WrightTools, and adds extra methods
* `MeshData.grad`: compute the gradient of a channel.  Gradient is useful for BZ integration procedures.
