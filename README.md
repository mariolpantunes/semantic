# DPWC Semantic Model

Distributional Profile of multiple Words Categories (DPWC) 
is a Distributional Profile model that uses clustering to split
the semantic dimensions based on their similarity.

The model also uses Matrix Factorization with latent dimension
reductions to solve the sparse word frequency matrix.


## Running unit tests

Several unit tests were written to validate some corner cases.
The unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Run the following commands to execute the unit tests.

```bash
python -m unittest
```

## Documentation

This library was documented using the google style docstring, it can be accessed [here](https://mariolpantunes.github.io/semantic/).
Run the following commands to produce the documentation for this library.

```bash
pip install pdoc3
pdoc -c latex_math=True --html -o docs . --force
```