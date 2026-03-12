import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Generate Synthetic Data
# -------------------------------

np.random.seed(42)

# Feature (X)
X = np.random.rand(100, 1)

# Target (y)
# True relationship: y = 4 + 3x + noise
y = 4 + 3 * X + np.random.randn(100, 1)

print("Sample X:", X[:5])
print("Sample y:", y[:5])


# -------------------------------
# Step 2: Initialize Parameters
# -------------------------------

w = np.random.randn(1)
b = np.random.randn(1)

learning_rate = 0.1
epochs = 1000

losses = []


# -------------------------------
# Step 3: Gradient Descent Training
# -------------------------------

for epoch in range(epochs):

    # Prediction
    y_pred = w * X + b

    # Error
    error = y_pred - y

    # Compute gradients
    dw = (2 / len(X)) * np.sum(error * X)
    db = (2 / len(X)) * np.sum(error)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Compute loss (MSE)
    loss = np.mean(error ** 2)
    losses.append(loss)

print("\nFinal weight (w):", w)
print("Final bias (b):", b)


# -------------------------------
# Step 4: Plot Training Loss
# -------------------------------

plt.figure()
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
#plt.show()


# -------------------------------
# Step 5: Plot Model Fit
# -------------------------------

plt.figure()
plt.scatter(X, y, label="Data")

y_line = w * X + b
plt.plot(X, y_line, color="red", label="Model")

plt.title("Linear Regression Fit")
plt.legend()

plt.show()