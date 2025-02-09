# Copyright 2025 Omar Huseynov
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <log_file>')
    sys.exit(0)

loss = []
accuracy = []
class_a_accuracy = []
class_b_accuracy = []
class_names = []

with open(sys.argv[1], 'r') as file:
    lines = file.readlines()

words = lines[0].split()
class_names = [words[10].strip(':').capitalize(), words[12].strip(':').capitalize()]

for line in lines:
    words = line.split()
    loss.append(float(words[3].strip(',%')))
    accuracy.append(float(words[9].strip(',%')))
    class_a_accuracy.append(float(words[11].strip(',%')))
    class_b_accuracy.append(float(words[13].strip(',%')))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(0, loss[0], label=f'Initial: {loss[0]:.4f}', color='green')
plt.scatter(len(loss) - 1, loss[-1], label=f'Final: {loss[-1]:.4f}', color='red')
plt.plot(loss, label='Loss', color='blue')
plt.xticks(range(0, 105, 5))
plt.yticks([i / 100 for i in range(0, 75, 5)])
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(0, accuracy[0], label=f'Initial: {accuracy[0]:.4f}%', color='green')
plt.scatter(len(accuracy) - 1, accuracy[-1], label=f'Final: {accuracy[-1]:.4f}%', color='red')
plt.plot(accuracy, label='Accuracy', color='orange')
plt.xticks(range(0, 105, 5))
plt.yticks(range(0, 105, 5))
plt.xlabel('Epoch')
plt.ylabel('Value (%)')
plt.title('Overall Accuracy')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(0, class_a_accuracy[0], label=f'Initial: {class_a_accuracy[0]:.4f}%', color='blue')
plt.scatter(len(class_a_accuracy) - 1, class_a_accuracy[-1], label=f'Final: {class_a_accuracy[-1]:.4f}%', color='blue')
plt.scatter(0, class_b_accuracy[0], label=f'Initial: {class_b_accuracy[0]:.4f}%', color='orange')
plt.scatter(len(class_b_accuracy) - 1, class_b_accuracy[-1], label=f'Final: {class_b_accuracy[-1]:.4f}%', color='orange')
plt.plot(class_a_accuracy, label=class_names[0])
plt.plot(class_b_accuracy, label=class_names[1])
plt.xticks(range(0, 105, 5))
plt.yticks(range(0, 105, 5))
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Class Accuracies')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

