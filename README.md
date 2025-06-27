# Mathrobo

Mathrobo is a lightweight library designed to support mathematical optimization and computations related to robotics.

## Installation

### Clone the Repository
Clone the repository to your local machine using:

```bash
git clone https://github.com/MathRobotics/MathRobo.git
```

### Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Install the Package
To install Mathrobo in your local environment, use:

```bash
pip install .
```

## Examples
Refer to the examples in the `examples` folder, where you can find Jupyter notebooks and scripts demonstrating various use cases of the library.

## Usage

Here is a quick example that computes the numerical gradient of a simple function:

```python
import numpy as np
import mathrobo as mr

f = lambda x: np.sum(x**2)
x = np.array([1.0, 2.0, -3.0])
grad = mr.numerical_grad(x, f)
print(grad)
```

You can also work with spatial transformations using the `SE3` class. The
following snippet creates a 90 degree rotation around the Z axis with a
translation, applies the transformation to a point and then inverts it:

```python
import numpy as np
import mathrobo as mr

# Rotation of 90 deg about Z and translation of 1 m along X
rot = mr.SO3.exp(np.array([0.0, 0.0, 1.0]), np.pi / 2)
T = mr.SE3(rot, np.array([1.0, 0.0, 0.0]))

point = np.array([0.0, 1.0, 0.0])
transformed = T @ point
recovered = T.inv() @ transformed

print(transformed)
print(recovered)
```

## Running Tests

To ensure Mathrobo is working correctly, run the tests using:

```bash
cd tests
pytest
```

## Contributing
Contributions are welcome! Feel free to report issues, suggest features, or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
