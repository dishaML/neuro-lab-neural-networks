Neuro Lab Neural Networks

A beginner-to-advanced neural network laboratory built from scratch in Python.
This project focuses on understanding the mathematics, architecture, and implementation of classic neural network models without relying heavily on high-level deep learning frameworks.

Features
1.Neural Networks implemented from scratch
2.Clean modular architecture
3.Educational and research-focused
4.Visualization-friendly structure
5.Multiple neural network models
6.Reusable utility modules
7.Test-driven development structure
8.Implemented / Planned Models


Model & Status
Perceptron	✅
Feed Forward Neural Network	🚧
Hopfield Network	🚧
Radial Basis Function Network (RBF)	🚧
Self Organizing Map (SOM)	🚧
Pattern Recognition Modules	🚧
Activation Functions Library	✅

Project Structure
Neuro-lab-neural-networks/
│
├── activation_function/
│   ├── activation.py
│   └── __init__.py
│
├── perceptron/
│   ├── perceptron.py
│   └── __init__.py
│
├── feedforward_network/
│   ├── feedforward.py
│   └── __init__.py
|
├── hopfield/
│   ├── hopfield.py
│   └── __init__.py
├── rbf_network/
│   ├── rbf.py
│   └── __init__.py
├── som/
│   ├── som.py
│   └── __init__.py
├── pattern_recognition/
│   ├── mist_demo.py
│   └── __init__.py
├── utils/
│   ├── helpers.py
│   └── __init__.py
├── tests/
│   ├── test_feedforward_train.py
│   └── __init__.py
|   ├── test_perceptron.py
|   ├── test_feedforward.py
|   ├── test_som.py
|   ├── test_hopfield.py
|   ├── test_boltzmann.py
|   └── __init__.py
|    ├── test_activation.py  
|    ├── test_perceptron.py  
|    
├── requirements.txt
├── README.md
└── .gitignore


Installation :
Clone the repository:
git clone https://github.com/dishaML/neuro-lab-neural-networks.git

Move into the project directory:
cd neuro-lab-neural-networks

Install dependencies:
pip install -r requirements.txt

Example Usage
Perceptron Example
from perceptron.perceptron import Perceptron

model = Perceptron(
    learning_rate=0.01,
    epochs=100
)

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 0, 0, 1]

model.fit(X, y)

prediction = model.predict([1, 1])

print("Prediction:", prediction)

Goals of This Project
Learn neural networks from first principles
Build ML intuition mathematically
Create reusable neural network implementations
Understand training algorithms deeply
Practice clean software engineering in AI projects
Develop a strong AI/ML portfolio project

Technologies Used:
1.Python
2.NumPy
3.Matplotlib
4.Scikit-learn (optional utilities/testing)
5.Future Improvements
6.Backpropagation engine
7.Model serialization
8.Training visualizations
9.GPU acceleration experiments
10.Advanced optimization algorithms
11.CNN and RNN implementations
12.Interactive demos
13.Contributing

Contributions, suggestions, and improvements are welcome.

Fork the repository
Create a feature branch
Commit your changes
Open a pull request


License
This project is licensed under the MIT License.

Author
Developed by Disha Das

GitHub Repository:
neuro-lab-neural-networks