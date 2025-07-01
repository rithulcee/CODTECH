from pulp import *

# Define problem
prob = LpProblem("Maximize_Profit", LpMaximize)

# Variables
x = LpVariable("Product_X", 0)
y = LpVariable("Product_Y", 0)

# Objective function
prob += 40*x + 30*y, "Total Profit"

# Constraints
prob += 2*x + y <= 100  # labor hours
prob += x + 2*y <= 80   # material limit

# Solve
prob.solve()

print("✅ Solution Status:", LpStatus[prob.status])
print("Product_X =", x.varValue)
print("Product_Y =", y.varValue)
print("Total Profit = ₹", value(prob.objective))
