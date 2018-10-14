[![Wisdomic Panda](https://github.com/robagwe/wisdomic-panda/blob/master/imgs/panda.png)](http://www.rohanbagwe.com/)  **Wisdomic Panda**
> *Hold the Vision, Trust the Process.*

# A beginner’s guide to understanding the Artificial Neural Network! 
*... simple intuition behind Artificial Neural Nets using plain python and numpy.*

![ANN](https://github.com/robagwe/wisdomic-panda/blob/master/imgs/NN.gif)

###### :exclamation: :clipboard: During my learning experience, I worked with neural nets for quite some time before formatting this write-up, and certainly it was the best investment of time as it helped me to understand how exactly The Vision (a character from MCU) was created after Tony Stark and Bruce Banner upload the J.A.R.V.I.S. into a synthetic body. :grin::grin: 

###### Thus, I’ve decided to build a Neural Network from scratch without any existing deep learning libraries like TensorFlow, Keras and many other. 

###### :exclamation: :clipboard: Although these libraries make it easy to build neural nets without fully understanding the inner workings of a Neural Network, I find that it’s beneficial for those who are starting fresh.
###### :exclamation: :clipboard: This is a very basic write-up that contains an ANN implementation from scratch. ANN (branch of Artificial Intelligence) is a deep ocean of mathematical computations and theories. And thus, I will go through the steps required for building a three layer neural network. I’ll go through a problem and explain you the process along with the most important concepts along the way. The only motivation is as a kick-off write-up, it must catalyst your interest in field of AI and get you ready to dive deep into the realms of learning. 

> *"Well begun is half done." - Aristotle*

###### Hopefully my learnings will be useful for you as well! So, Let’s get started!:boom:


### What is a Neural Network?

:bulb: *Human brain is one of the powerful learning mechanisms on the planet. The whole purpose of ANN is to mimic how the brain functions. So we want to create an infrastructure for machines to learn as humans do!*

:bulb: *The Neuron is the basic unit of computation in the brain, it receives and integrates chemical signals from other neurons and depending on a number of factors it either does nothing or generates an electrical signal or Action Potential which in turn signals other connected neurons via synapses.*

**A typical biological neural network looks like this:**

![BNN](https://github.com/robagwe/wisdomic-panda/blob/master/imgs/BNN.png)

Artificial Neural Networks is a term derived from Biological neural networks that construct the structure of a human brain. Like the human brain has neurons interconnected to each other, artificial neural networks also has neurons that are interconnected to each other in different layers of the network. These neurons are called as nodes. A network is a collection of interconnected nodes.

**A typical artificial neural network looks something like this:**
![BNN](https://github.com/robagwe/wisdomic-panda/blob/master/imgs/ANN.png)

:bulb: *An artificial neural network is essentially a computational network based on biological neural networks. These models aim to duplicate the complex network of neurons in our brains.So this time, the nodes are programmed to behave like actual neurons. Although they’re really artificial neurons that try to behave like real ones, hence the name “Artificial Neural Network."*

:bulb: *There are input layer neurons (circles in yellow): In analogy  of your brain input layer neurons are basically your sensors: see, hear, feel, touch, smell, taste! Dendrites from Biological Neural Network represents Inputs in Artificial Neural Network, Cell Nucleus represents Nodes, Synapse represents Weights and Axon represents Output. Synapses are assigned weights : and this is how NN learns by adjusting the weights, what signal is important and what is not … so when you train a NN you basically adjust the weights so that they become ready to get the real time inputs and based on the past experience it will generate the output.*


:bulb: *Dreams,memories,ideas,self regulated movement, reflexes and everything you think or do is all generated through this process: millions, maybe even billions of neurons firing at different rates and making connections which in turn create different subsystems all running in parallel and creating a biological Neural Network.With approximately 100 billion neurons, the human brain processes data at speeds as fast as 268 mph! In essence, a neural network is a collection of neurons connected by synapses. This collection is organized into three main layers: the input layer, the hidden layer, and the output layer. You can have many hidden layers, which is where the term deep learning comes into play. In an artifical neural network, there are several inputs, which are called features, and produce a single output, which is called a label.*

### The problem to solve ###

> Cristiano Ronaldo is now 32 years of age and, according to many, Real Madrid star Cristiano Ronaldo hailed as a footballing legend. At 32, Ronaldo has already done it all. Recently, Cristiano Ronaldo Transfered to Juventus from Real Madrid for Reported £105M Fee, Which was super cool. But there were many who tweeted "CR7, Is he really worth it? :unamused::unamused: " 

> Well, *Haters keep on hating, cause somebody's gotta do it.* So, :unamused:! At 32 years of age, Ronaldo's time at the top of the game could be limited, but his impact in recent years has increased rather than diminished with age. Thus, I decided to implement a simple artificial neural network that predicts CR7's performace in upcoming years. As a CR7 fan, I have extracted this data from internet so that our artificial brain understands and learns the relation between different features mentioned below and predicts the number of goals Ronaldo can score at his current age. 


Our ANN will model a single hidden layer with five inputs and one output. In the network, we will be predicting the goals scored by CR based on the inputs of his **Total Footbala Appearances**, **Playing Time in minutes**, **Yellow Cards** and **Red Cards** at his given **Age**. The number of **Goals** Scored the output.

### [Oracle_of_Football](https://github.com/robagwe/kick-off-Oracle_of_Football-The_First_Artificial_Neural_Network/blob/master/code/Oracle_of_Football-The_First_Artificial_Neural_Network.py)

###### Here’s our sample data of what we’ll be training our Neural Network on: ######


| Age | Total Appearance(s) | Playing Time (min) | Yellow Card | Red Card | Goals|
| :---         |     :---:      |    :---:      |   :---:       |   :---:       |   :---:       |
| 24   | 37    | 3290    | 5 | 1 | 34|
| 25   | 52    | 4653    | 5 | 0 | 53|
| 26   |56| 4653| 5| 0| 53 |
| 27   |57| 5009| 13| 0| 59|
| 28   |54| 4794| 7| 1| 65|
| 29   |48| 3290| 5| 1| 51|
| 30   |57| 5243| 6| 0| 56|
| 31   |44| 4893| 3| 1| 48|
| 32   |40| 3000| 0| 0| :question: |


### Understanding the process ###

###### ANN consists of 3 major layers:

- **Input layer** - Accepts inputs in several different formats.
- **Hidden Layers** - Performs all the calculations and manipulations to extract hidden features and patterns.
- **Output Layer** - Produces the desired output.

> This calculated weighted sum is passed an input to an activation function to generate the output. Activation functions decide whether a node should be fired or not. Only those which are fired make it to the output layer. There are different activation functions available that can be applied depending on the kind if task you are performing.

###### ANN ingredients:

- An input layer, x
- An arbitrary amount of hidden layers
- An output layer, y
- A set of weights and biases between each layer, W and b
- A choice of activation function for each hidden layer,s. In this tutorial, we’ll use a Sigmoid activation function.

###### Here’s a brief overview of how a simple neural network works: ######

1. **Randomly initialise the weights to small numbers close to 0 (but not 0).**
2. **Input the first observation of your dataset in the input layer, each feature in one input node.**
3. **Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neurons's activation is limited by the weights. Propagate the activations until getting the predicted result.**
4. **Compare the predicted result to the accurate result. Measure the Error.**

5. **Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.**
6. **Repeat steps 1 to 5 and update the weights after each observation.(Reinforcement Learning). OR
Repeat steps 1 to 5 and update the weights only after a batch of observations.(Batch Learning).**
7. **When the whole training set passed through the ANN, that makes an epoch. Redo more Epochs.**

:bulb: *This is also called as Gradient Descent Algorithm.*

###### Forward Propagation: ######
      
      
      def forward_Propogation(self, X):
        
        self.hl_input = np.dot(X, self.w_hidden) 
        # (8*5) dot (5*3) = (8*3)
        self.hl_input = self.hl_input + self.b_hl
        # (8*3) + (1*3)
        self.hl_activations = self.sigmoid(self.hl_input)
        
        self.ol_input = np.dot(self.hl_activations, self.w_out) # 8*3 dot 3*1 = 8 * 1
        self.ol_input = self.ol_input + self.b_ol
        output = self.sigmoid(self.ol_input)
        return output # or ol_activation
    
   
###### Backward Propagation: ######

      def backward_Propogation(self, X, Y, output):

        E = Y-output
        print("Error Calculated: ",E)
        slope_ol = self.sigmoidPrime(output)
        slope_hl = self.sigmoidPrime(self.hl_activations)

        delta_ol = E * slope_ol
        Error_at_hidden_layer = delta_ol.dot(self.w_out.T)
        delta_hl = Error_at_hidden_layer * slope_hl

        # Update weight at both output and hidden layer

        self.w_out += self.hl_activations.T.dot(delta_ol) * self.learning_rate
        self.b_ol += np.sum(delta_ol, axis=0,keepdims=True) * self.learning_rate
        self.w_hidden += X.T.dot(delta_hl) * self.learning_rate
        self.b_hl += np.sum(delta_hl, axis=0,keepdims=True) * self.learning_rate

###### Notes: ######

> :pushpin: **Activation Function**: 
This calculated weighted sum is passed an input to an activation function to generate the output. Activation functions decide whether a node should be fired or not. Only those which are fired make it to the output layer. We need activation function to cope with the highly non linear feature of the input data set. There are different activation functions available that can be applied depending on the kind if task you are performing like: “Sigmoid”, “Tanh”, ReLu and many other.
 

> :pushpin: **Use of derivative**:
When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.That is why we use differentiation in almost every part of Machine Learning and Deep Learning.

> :pushpin: **Epochs**:
One forward and backward propagation iteration is considered as one training cycle or one epoch. As I mentioned earlier, When do we train second time then update weights and biases are used for forward propagation.




### Final Thoughts ###

> We did it! Our feedforward and backpropagation algorithm trained the Neural Network successfully and the predictions converged on the true values. Note that there’s a slight difference between the predictions and the actual values. This is desirable, as it prevents overfitting and allows the Neural Network to generalize better to unseen data.

> At its core, neural networks are simple. They just perform a dot product with the input and weights and apply an activation function. When weights are adjusted via the gradient of loss function, the network adapts to the changes to produce more accurate outputs.



### My Recommendation: ###
:speech_balloon:*Neural networks, roughly simulate the working of Human brain to make machines learn from examples. Given a large dataset, ANN learns a mapping between inputs and outputs, and thus it can predict outputs for unseen inputs. But at it’s core it is just a Mathematical Optimization. Even though it posses an ability to make decisions and classify datasets, it is very narrow in the way it works. This is not what our brain actually does, our brain doesn’t require 10000 images of cats to recognize a panda.
Fortunately for us, our journey isn’t over. There’s still much to learn about Neural Networks and Deep Learning. There're so many interesting algorithms such as ANN, RNN, CNN and concepts such as Exploding Gradient, Vanishing Gradient and different Activation Functions.*

:speech_balloon:*If you're serious about Neural Nets then try to rebuild this network from scratch by forming dataset of your own interst. Trust me! It seriously helps. If you want to be able to create models based on new academic papers or read and understand sample code for these different ANN models, I think you get your hands on as soon as possible. I think it's useful even if you're using frameworks like Tensorflow, Keras, pyTorch or Sklearn.*

## <img src="https://github.com/robagwe/wisdomic-panda/blob/master/imgs/acr.png" width="50">   Hey Buddy!</img>

> This repository explains the rationale for Neural Networks in python. I have implemented a very basic example using core python, please have a dekko at it! This repo covers approximately 1% of the entire Neural Networks.
If you have any suggestions for more concepts that should be on this page or you notice a mistake, please let me know or consider submitting a pull request so others can benefit from your work. 
Thank you very much for reaching out! Please follow if you find it handy and hit :star: to get more kick-off repo updates.

:email: [Drop In!!](https://www.rohanbagwe.com) Seriously, it'd be great to discuss Technology.

>*"Everything you can imagine is real." - Pablo Picasso*
