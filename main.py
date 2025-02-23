
import numpy as np

# Load sinogram
y = np.load(r"C:\Users\dqz\Downloads\sinogram_phantom.npy")

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import matplotlib.pyplot as plt

# Define parameters
num_views, num_det_rows, num_det_channels = y.shape
angles = np.linspace(-np.pi / 2, np.pi / 2, num_views)  # Convert to NumPy


# Initialize MBIR-JAX model
sinogram_shape = (num_views, num_det_rows, num_det_channels)
ct_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)


# Initialize reconstruction volume
x_init = np.zeros((num_det_channels, num_det_channels, num_det_rows))


# Define loss function
def loss_fn(x):
   Ax = ct_model.forward_project(jnp.asarray(x))  # Ensure input is JAX-compatible
   M = y.size  # Total number of elements in y
   return jnp.sum((y - Ax) ** 2) / M  # Compute normalized loss


# Compute gradient using JAX autograd
grad_loss_fn = jax.grad(loss_fn)


def gradient_descent_step(x, lr):
   grad = grad_loss_fn(x)  # Compute gradient
   return x - lr * grad  # Update step


# Gradient descent parameters
num_iterations = 2000
learning_rate = 0.5
x = x_init  # Initialize x


# Run gradient descent
print("Starting MBIR Reconstruction...")
for i in range(num_iterations):
   x = gradient_descent_step(x, learning_rate)


   if i % 10 == 0:
       loss = loss_fn(x)
       print(f"Iteration {i}: Loss = {loss:.6f}")


# Convert to NumPy for visualization
mbir_reconstruction = np.array(x)


# Display Results
plt.figure(figsize=(12, 6))
plt.imshow(mbir_reconstruction[:, :, mbir_reconstruction.shape[2] // 2], cmap='gray')
plt.title("MBIR Reconstruction (JAX Autograd)")
plt.axis("off")
plt.show()


# Plot all slices
num_slices = mbir_reconstruction.shape[2]
cols = 8  # Number of columns in the grid
rows = int(np.ceil(num_slices / cols))  # Compute number of rows

fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
for i in range(num_slices):
    row, col = divmod(i, cols)
    axes[row, col].imshow(mbir_reconstruction[:, :, i], cmap='gray')
    axes[row, col].axis("off")
    axes[row, col].set_title(f"Slice {i}")

# Hide unused subplots
for i in range(num_slices, rows * cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()
