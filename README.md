# Machine Learning Assignment 2
## Implementing Back-Propagation Algorithm
### Submitted by: Shashwat Kadam (BT16CSE035)

**Problem statement:** Implement a back propagation algorithm using any programming language

<h1 style='font-family:georgia; font-size:28px'>1. Structure of the program</h1>

<p style='font-family:georgia;'> The program is written in Python programming language. Python has numerous supporting libraries like 'NumPy', 'Pandas'm etc., which can make many mathematical operations efficient and avoid unwanted iterations.</p>

<p style='font-family:georgia;'> The program is written in such a way that, you have to make as less changes as possible to customize the neural the neural network according to your need</p>

<p style='font-family:georgia;'> You can create your own network structure in JSON format (some sample files are provided), add required no. of neurons, no. of layers, your activation function of choice, <b>anything you want in the JSON file and the Neural Network will be built and initialized according to it</b>. All you need to do is while creating the Network, just pass the file-path of the network structure JSON file.</p>

<p style='font-family:georgia;'>The Code has three important modules:</p>
<ul style='font-family:georgia;'> 
    <li> <b>BackProp.py</b>: All the algorithms, data structures, class definitions are present here </li>
    <li> <b>Runner.py</b>: A module to run the algorithm </li>
    <li> <b>activation_functions.py</b>: A module containing various different activation function definitions, e.g., sigmoid(), tanh(), relu(), etc.
</ul>
<p style='font-family:georgia;'>Now let's understand the Data structures and Class definitions</p>

<h1 style='font-family:georgia; font-size:28px'> 2. Data Structures</h1>

<p style='font-family:georgia;'>The three important data structures created for the program are viz., 'Neuron', 'Layer', and 'NeuralNetwork'.</p>

<h2 style='font-family:georgia; font-size: 24px'>2.1. class Neuron:</h2>

<img src='./images/Neuron.png'/>
<p style='font-family:georgia;'>Some text describing the Neuron class</p>

<h2 style='font-family:georgia; font-size: 24px'>2.2. class Layer:</h2>

<img src='./images/Layer.png'/>
<p style='font-family:georgia;'>Some text describing the Layer class</p>

<h2 style='font-family:georgia; font-size: 24px'>2.3. class NeuralNetwork:</h2>

<img src='./images/NeuralNetwork.png'/>
<p style='font-family:georgia;'>Some text describing the NeuralNetwork class</p>