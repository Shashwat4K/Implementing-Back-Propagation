# Machine Learning Assignment 2
## Implementing Back-Propagation Algorithm
### Submitted by: Shashwat Kadam (BT16CSE035)
<hr/>

**Problem statement:** Implement a back propagation algorithm using any programming language

<h1 style='font-family:georgia; font-size:28px'> Contents</h1>
<ul style='font-family:georgia;'>
    <li>   1. Structure of the program
    <li>   2. Data Structures
        <ul>
            <li> 2.1. class Neuron
            <li> 2.2. class Layer
            <li> 2.3. class NeuralNetwork
        </ul>
</ul>    
<hr/>


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
<p style='font-family:georgia;'>A Neuron is the basic unit of every Network layer. It is a simple unit which is connected to all the previous layer neurons and the next layer with appropriate edge weights. It has various attributes like Activation value (value after applying activation function to the weighted sum), Activation function (function to be applied to the weighted sum of inputs and corresponding weights), error term (value of error term 'delta'), and Neuron type (Input layer neuron or other). A neuron has following methods:</p>
<ul style='font-family:georgia;'>
    <li><b>activate_neuron(input_vector, weights, bias_value)</b>: Depending on the type of neuron (input or hidden or output) it calculates weighted sum of 'input_vector' and 'weights' (Using np.dot()) and applies the activation function e.g. sigmoid() to the result. The final result is stored in Neuron's 'activation' attribute.</li>
    <li><b>Setters and Getters</b></li>
</ul>

<h2 style='font-family:georgia; font-size: 24px'>2.2. class Layer:</h2>

<img src='./images/Layer.png'/>
<p style='font-family:georgia;'>A 'Layer' object is a collection of 'Neuron' objects, weight matrix between current layer and 'previous' layer, and various static layer attributes. Following are the 'Layer' attributes:</p>
<ul style='font-family:georgia;'>
    <li><b>layer_type</b>: Type of layer. Input, Hidden or Output ('i', 'h', 'o' respectively).</li>
    <li><b>num_units</b>: No. of neurons in the layer.</li> 
    <li><b>prev_layer_neurons</b>: No. of neurons in previous layer (used to initialize weight matrix).</li>
    <li><b>neurons</b>: List of Neurons for this layer </li>
    <li><b>weight_matrix</b>: The weight matrix between current layer object and previous layer object.
    <li>Other static attributes like <b>layer_ID</b>, etc.</li> 
</ul>
<p style='font-family:georgia;'>A layer has crucial auxilliary functions which are useful in forward and backward propagation. Following are some important layer methods:</p>
<ul style='font-family:georgia;'>
    <li><b>get_neuron_activations()</b>: Returns a 1D numpy vector with all the activation values of the neurons.</li>
    <li><b>get_neuron_error_terms()</b>: Returns a 1D numpy vector with all the error term values of the neurons.</li>
    <li><b>get_weights_for_neuron(index)</b>: Gives a vector of weights connected to the particular neuron at position 'index' in the list of neurons.</li>
    <li><b>calculate_error_terms(is_output, resource)</b>: Calculate and set the values of error term for every neuron in the layer (depending the value of 'is_output').If 'is_output' = True then calculate according to output layer otherwise calculate according to hidden layer. 'resource' is the packet containing data which will be used to calculate the error term values.</li>
    <li><b>update_weights(X, DELTA, learning_rate)</b>: Updates the weight matrix (used in backward pass). 'X' is the input vector to the layer and 'DELTA' is the vector of error terms.</li>
    <li>Other setters and getters</li>
</ul>

<h2 style='font-family:georgia; font-size: 24px'>2.3. class NeuralNetwork:</h2>

<img src='./images/NeuralNetwork.png'/>
<p style='font-family:georgia;'>NeuralNetwork is the final structure consisting of the list of layers. All the training process takes place here. The network is costructed from the JSON file which is supplied while creating an instance of this class.</p>