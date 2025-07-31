# my_scripts

**my_scripts** is a curated collection of general-purpose Python scripts and tools developed to support a wide range of scientific computing, data analysis, and automation tasks. This repository emphasizes reusability, modular design, and clarity—making it a solid toolbox for researchers, analysts, and developers working with Python.

---

## 📁 Repository structure

### `Core_tools/`

This is the core of the repository. It includes foundational Python modules grouped by functionality:

- **`basic.py`**  
  A collection of frequently used utility functions for file handling, string manipulation, parsing, numerical operations, and general scripting tasks.

- **`core_plots.py`**  
  A lightweight plotting library built on `matplotlib`, providing functions for quick and customizable data visualization.

- **`coretools.py`**  
  Contains advanced utility classes and abstractions, including the `Result` class, which acts both as a dictionary and a class instance—useful for returning structured results (similar to `scipy.optimize.OptimizeResult`).

- **`statistics.py`**  
  Implements statistical functions and algorithms, including Metropolis Monte Carlo sampling and statistical error analysis tools.

➡️ See the [`Core_tools_examples`](Core_tools_examples) directory for scripts that demonstrate usage of these modules.

---

## 📦 Additional projects & notebooks

The repository also includes various standalone or semi-standalone utilities:

- [`Demuxing`](Demuxing): Tools for demultiplexing structured data streams or files.
- [`statistical_error_analysis/`](statistical_error_analysis): Scripts and functions for quantifying and visualizing uncertainty in Molecular Dynamics (MD) simulations (boostrap and block-analysis methods).
- [`arithm_vs_geom_growth.ipynb`](arithm_vs_geom_growth.ipynb): A Jupyter notebook comparing arithmetic and geometric growth models with mathematical insight and plots.
- [`kish_size.ipynb`](kish_size.ipynb): A notebook for computing Kish effective sample size, commonly used in survey sampling and weight analysis.
- [`Instructions/`](Instructions/): Documentation, one-off notes, or usage instructions for specialized cases or helper scripts.

---

## 🧪 Requirements & setup

Most tools rely on standard Python 3 libraries. You may need the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `tqdm`

or others, depending on your usage.

Install dependencies with:

```
pip install -r requirements.txt
```

---

## 🚀 Usage Examples
Example scripts in `Core_tools_examples` demonstrate how to use the tools. For instance:

```
from Core_tools.basic import flatten
from Core_tools.coretools import Result
from Core_tools.statistics import metropolis_sampler

# Use a basic function
flat_list = flatten([[1, 2], [3, 4]])

# Use Result as a hybrid dictionary/object
res = Result(success=True, message="Done")
print(res.success)  # Object-style
print(res["message"])  # Dict-style
```

Run example scripts or explore notebooks interactively:

```
python Core_tools_examples/example_basic.py
jupyter notebook kish_size.ipynb
```

---

## 🤝 Contributing
You're welcome to fork the repository, report issues, or submit pull requests. Whether it's bug fixes, new utility functions, or notebooks demonstrating use cases—contributions are appreciated!

---

## 🔗 Related
If you use tools like `scipy`, `numpy`, or `matplotlib`, you'll find these scripts easy to integrate into your workflow. The design philosophy of `my_scripts` is lightweight, practical, and extensible.

