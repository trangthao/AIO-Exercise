##### EXERCISE 1 ####
#The code below shows the solution for exercise 1 of the AIO 2024 for Basic Python dated 1 June, 2024
def evaluate_classification_f1score(tp, fp, fn):
    # Check if the inputs are integers
    if not isinstance(tp, int):
        print("tp must be int")
        return
    if not isinstance(fp, int):
        print("fp must be int")
        return
    if not isinstance(fn, int):
        print("fn must be int")
        return
    
    # Check if the inputs are greater than zero
    if tp <= 0 or fp <= 0 or fn <= 0: 
        print("tp and fp and fn must be greater than zero")
        return
    
    # Calculate Precision, Recall, and F1-Score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the results
    print(f"precision is {precision}")
    print(f"recall is {recall}")
    print(f"f1-score is {f1_score}")

# Examples
evaluate_classification_f1score(tp=2, fp=3, fn=4)
evaluate_classification_f1score(tp='a', fp=3, fn=4)
evaluate_classification_f1score(tp=2, fp='a', fn=4)
evaluate_classification_f1score(tp=2, fp=3, fn='a')
evaluate_classification_f1score(tp=2, fp=3, fn=0)
evaluate_classification_f1score(tp=2.1, fp=3, fn=0)

##### EXERCISE 2 ####
#The code below shows the solution for exercise 2 of the AIO 2024 for Basic Python dated 1 June, 2024
import math

def is_number(n):
    try: 
        float(n)
        return True
    except ValueError:
        return False

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def elu(x, alpha=0.01):
    return x if x > 0 else alpha * (math.exp(x) - 1)

def activate_function():
    x = input("Input x = ")
    if not is_number(x):
        print("x must be a number")
        return
    
    x = float(x)

    activation_function = input("Input activation function (sigmoid|relu|elu): ").lower()

    if activation_function == "sigmoid":
        result = sigmoid(x)
        print(f"sigmoid({x}) = {result}")
    elif activation_function == "relu":
        result = relu(x)
        print(f"relu({x}) = {result}")
    elif activation_function == "elu":
        result = elu(x)
        print(f"elu({x}) = {result}")
    else: 
        print(f"{activation_function} is not supported")

# Examples
activate_function()

##### EXERCISE 3 ####
#The code below shows the solution for exercise 3 of the AIO 2024 for Basic Python dated 1 June, 2024
import random
import math

def is_integer(n):
    return n.isdigit()

def calculate_loss(num_samples, loss_name):
    samples = list(range(1, num_samples + 1))
    predictions = [random.uniform(0, 10) for _ in range(num_samples)]
    targets = [random.uniform(0, 10) for _ in range(num_samples)]
    
    if loss_name == "MAE":
        losses = [abs(pred - target) for pred, target in zip(predictions, targets)]
        loss_value = sum(losses) / num_samples
    elif loss_name == "MSE":
        losses = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
        loss_value = sum(losses) / num_samples
    elif loss_name == "RMSE":
        losses = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
        loss_value = math.sqrt(sum(losses) / num_samples)
    else:
        print(f"{loss_name} is not supported")
        return
    
    for i in range(num_samples):
        print(f"loss name: {loss_name}, sample: {samples[i]}, pred: {predictions[i]}, target: {targets[i]}, loss: {losses[i]}")
    print(f"final {loss_name}: {loss_value}")

def reg_loss_function():
    num_samples = input("Input number of samples (integer number) which are generated: ")
    if not is_integer(num_samples):
        print("number of samples must be an integer number")
        return
    
    num_samples = int(num_samples)
    
    loss_name = input("Input loss name (MAE, MSE, RMSE): ").upper()
    
    calculate_loss(num_samples, loss_name)

# Example usage
reg_loss_function()

##### EXERCISE 4 ####
#The code below shows the solution for exercise 4 of the AIO 2024 for Basic Python dated 1 June, 2024
def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def approx_sin(x, n):
    sin_approx = 0
    for i in range(n):
        term = ((-1)**i) * (x**(2*i + 1)) / factorial(2*i + 1)
        sin_approx += term
    return sin_approx

def approx_cos(x, n):
    cos_approx = 0
    for i in range(n):
        term = ((-1)**i) * (x**(2*i)) / factorial(2*i)
        cos_approx += term
    return cos_approx

def approx_sinh(x, n):
    sinh_approx = 0
    for i in range(n):
        term = (x**(2*i + 1)) / factorial(2*i + 1)
        sinh_approx += term
    return sinh_approx

def approx_cosh(x, n):
    cosh_approx = 0
    for i in range(n):
        term = (x**(2*i)) / factorial(2*i)
        cosh_approx += term
    return cosh_approx

# Example usage
x = 3.14
n = 10

print(f"approx_sin(x={x}, n={n}) = {approx_sin(x, n)}")
print(f"approx_cos(x={x}, n={n}) = {approx_cos(x, n)}")
print(f"approx_sinh(x={x}, n={n}) = {approx_sinh(x, n)}")
print(f"approx_cosh(x={x}, n={n}) = {approx_cosh(x, n)}")

##### EXERCISE 5 ####
#The code below shows the solution for exercise 5 of the AIO 2024 for Basic Python dated 1 June, 2024
def mean_dif_nroot_error(y, y_hat, n, p):
    # Calculate the mean difference of n-th root error
    term = abs((y**(1/n)) - (y_hat**(1/n)))**p
    return term

# Example usage
y = float(input("Enter the value of y: "))
y_hat = float(input("Enter the value of y_hat: "))
n = int(input("Enter the value of n: "))
p = int(input("Enter the value of p: "))

result = mean_dif_nroot_error(y, y_hat, n, p)
print(f"MD_nRE for y={y}, y_hat={y_hat}, n={n}, p={p} is {result}")
