import numpy as np
#Input features
x1 = np.array([1, 2, 3, 4, 5], dtype=float)
x2 = np.array([60, 65, 70, 75, 85], dtype=float)
y = np.array([0, 0, 1, 1, 1], dtype=float)
n = len(y)

#Feature Scaling
x1_mean = x1.mean()
x1_std = x1.std()

x2_mean = x2.mean()
x2_std = x2.std()

x1 = (x1 - x1_mean) / x1_std
x2 = (x2 - x2_mean) / x2_std

#Helper functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, y_pred):
    y_pred = min(max(y_pred, 1e-15), 1 - 1e-15) #where le-15 is the smallest value but not zero (0.0000001) ,1 - 1e-15 highest value but not 1 (0.999)
    if y == 1:
        return -np.log(y_pred)
    else:
        return -np.log(1 - y_pred)

#Initial parameters
m1 = 0.0
m2 = 0.0
c = 0.0

learning_rate = 0.01
epochs = 100

#Training loop

for epoch in range(1, epochs + 1):
    total_loss = 0
    
    print(f"\n======== EPOCH {epoch} ========")
    print("Sample | x1 | x2 | y | z | y_pred | loss | m1 | m2 | c")
    print("-------------------------------------------------------")

    for i in range(n):
        X1 = x1[i]
        X2 = x2[i]
        y_i = y[i]

        #Forward
        z = m1 * X1 + m2 * X2 + c
        y_pred = sigmoid(z)

        #loss
        loss = log_loss(y_i, y_pred)
        total_loss += loss

        #Gradients
        error = y_pred - y_i
        m1 -= learning_rate * error * X1
        m2 -= learning_rate * error * X2
        c -= learning_rate * error

        # Print row
        print(f"{i+1:^6} | {X1:>7.3f} | {X2:>7.3f} | {int(y_i):^1} | "
              f"{z:>7.3f} | {y_pred:>6.3f} | {loss:>6.3f} | "
              f"{m1:>6.3f} | {m2:>6.3f} | {c:>6.3f}")
        
        print("------------------------------------------------------------------")
        print(f"Epoch {epoch} Average Cost = {total_loss / n:.4f}")

# Final results
print("\nFinal Model Parameters:")
print(f"m1 = {m1:.4f}")
print(f"m2 = {m2:.4f}")
print(f"c = {c:.4f}") 

print("\nFinal Predictions:")
print("Sample | y_actual | y_pred | class")
print("------------------------------------")
for i in range(n):
    z = m1 * x1[i] + m2 * x2[i] + c
    y_pred = sigmoid(z)
    cls = 1 if y_pred >= 0.5 else 0
    print(f"{i+1:^6} | {int(y[i]):^8} | {y_pred:>6.3f} | {cls:^5}")

#user input prediction (correct)
print("\n----- USER INPUT PREDICTION -----")

user_x1 = float(input("Enter Study Hours: "))
user_x2 = float(input("Enter Attendance (%):"))

#Scale Only user input using Training mean & std
user_x1_scaled = (user_x1 - x1_mean) / x1_std
user_x2_scaled = (user_x1 - x2_mean) / x2_std

z_user = m1 * user_x1_scaled + m2 * user_x2_scaled + c
y_pred_user = sigmoid(z_user)

user_class = 1 if y_pred_user >= 0.5 else 0

print("\nPrediction Result:")
print(f"Prediction Probability (Pass) = {y_pred_user:.4f}")

if user_class == 1:
    print("Final Decision: PASS")
else:
    print("Final Decision: FAIL")
