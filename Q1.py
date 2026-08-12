import numpy as np
import time
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes()

X = data.data
y = data.target

# Add bias column
X = np.column_stack((np.ones(X.shape[0]), X))

print("Dataset : Diabetes")
print("Samples :", X.shape[0])
print("Features:", X.shape[1] - 1)
print()


# Loss function
def loss(X, y, theta):
    prediction = X @ theta
    return np.mean((prediction - y) ** 2)


# -------------------------------------------------
# Batch Gradient Descent
# -------------------------------------------------

def batch_gd(X, y):

    n, m = X.shape
    theta = np.zeros(m)

    lr = 0.1
    epochs = 1000
    updates = 0

    start = time.time()

    for epoch in range(epochs):

        prediction = X @ theta

        gradient = (X.T @ (prediction - y)) / n

        theta = theta - lr * gradient

        updates += 1

    end = time.time()

    return end - start, updates, loss(X, y, theta)


# -------------------------------------------------
# Stochastic Gradient Descent
# -------------------------------------------------

def sgd(X, y):

    n, m = X.shape
    theta = np.zeros(m)

    lr = 0.01
    epochs = 1000
    updates = 0

    start = time.time()

    for epoch in range(epochs):

        for i in range(n):

            prediction = X[i] @ theta

            gradient = (prediction - y[i]) * X[i]

            theta = theta - lr * gradient

            updates += 1

    end = time.time()

    return end - start, updates, loss(X, y, theta)


# -------------------------------------------------
# Mini-Batch Gradient Descent
# -------------------------------------------------

def mini_batch_gd(X, y):

    n, m = X.shape
    theta = np.zeros(m)

    lr = 0.05
    epochs = 1000
    batch_size = 64
    updates = 0

    start = time.time()

    for epoch in range(epochs):

        for i in range(0, n, batch_size):

            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            prediction = X_batch @ theta

            gradient = (
                X_batch.T @ (prediction - y_batch)
            ) / len(X_batch)

            theta = theta - lr * gradient

            updates += 1

    end = time.time()

    return end - start, updates, loss(X, y, theta)


# -------------------------------------------------
# Train all models
# -------------------------------------------------

batch = batch_gd(X, y)
stochastic = sgd(X, y)
mini = mini_batch_gd(X, y)


# -------------------------------------------------
# Display results
# -------------------------------------------------

print("-" * 60)
print("Optimizer\tTime(s)\tUpdates\t\tFinal Loss")
print("-" * 60)

print(
    f"Batch GD\t{batch[0]:.4f}\t"
    f"{batch[1]}\t\t{batch[2]:.2f}"
)

print(
    f"SGD\t\t{stochastic[0]:.4f}\t"
    f"{stochastic[1]}\t\t{stochastic[2]:.2f}"
)

print(
    f"Mini-Batch\t{mini[0]:.4f}\t"
    f"{mini[1]}\t\t{mini[2]:.2f}"
)

print("-" * 60)


# -------------------------------------------------
# Conclusion
# -------------------------------------------------

print("\nConclusion:")
print("Mini-Batch Gradient Descent is generally suitable")
print("for large-scale datasets because it provides a")
print("good balance between computation and convergence.")
