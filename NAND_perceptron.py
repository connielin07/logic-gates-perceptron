import numpy as np

class Perceptron:    
    def __init__(self, input_length, weights=None, bias=None):
        if weights is None:
            self.weights = np.zeros(input_length)
        else:
            self.weights = weights
        if bias is None:
            self.bias = 0
        else:
            self.bias = bias    
    
    @staticmethod    
    def activation_function(x):
        if x >= 0:
            return 1
        return 0
        
    def __call__(self, input_data):
        weighted_input = self.weights * input_data
        weighted_sum = weighted_input.sum() + self.bias
        return Perceptron.activation_function(weighted_sum)
    
    def train(self, training_data, targets, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for x, target in zip(training_data, targets):
                prediction = self(x)
                error = target - prediction
                if error != 0:
                    self.weights += learning_rate * error * x
                    self.bias += learning_rate * error
                    errors += 1
            if errors == 0:
                print(f"訓練在第 {epoch+1} 輪收斂")
                break
        else:
            print("訓練未收斂")

# 建立訓練資料
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([1, 1, 1, 0])

# 建立感知器（初始權重為0）
NAND_Gate = Perceptron(2)

# 訓練模型
NAND_Gate.train(training_inputs, targets)

# 驗證模型
print("最終權重:", NAND_Gate.weights)
print("最終偏差:", NAND_Gate.bias)
print("預測結果:")
for x, expected in zip(training_inputs, targets):
    predicted = NAND_Gate(x)
    print(f"{x} -> Predicted: {predicted}, Expected: {expected}")