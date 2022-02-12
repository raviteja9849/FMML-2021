# FMML[Foundations in Modern Machine Learning]-2021

![image](https://user-images.githubusercontent.com/66083579/153562815-2d778c69-dcdf-492d-a5e5-8cf259883029.png)

# What is Machine Learning?

Machine Learning is an Application of Artificial Intelligence (AI) it gives devices the ability to learn from their experiences andimprove their self without doing any coding. 

![image](https://user-images.githubusercontent.com/66083579/153563637-fb3596dc-e00b-4bde-859a-6732050bd626.png)


Machine Learning is a subset of Artificial Intelligence. Machine Learning is the study of making machines more human-like in their behaviour and decisions by giving them the ability to learn and develop their own programs. This is done with minimum human intervention, i.e., no explicit programming. The learning process is automated and improved based on the experiences of the machines throughout the process. Good quality data is fed to the machines, and different algorithms are used to build ML models to train the machines on this data. The choice of algorithm depends on the type of data at hand, and the type of activity that needs to be automated. 

Now you may wonder, how is it different from traditional programming? Well, in traditional programming, we would feed the input data and a well written and tested program into a machine to generate output. When it comes to machine learning, input data along with the output is fed into the machine during the learning phase, and it works out a program for itself. To understand this better, refer to the illustration below:

![image](https://user-images.githubusercontent.com/66083579/153564159-833c014d-0633-405a-8e55-648dc2d892cf.png)


# Why Should We Learn Machine Learning?

Machine Learning today has all the attention it needs. Machine Learning can automate many tasks, especially the ones that only humans can perform with their innate intelligence. Replicating this intelligence to machines can be achieved only with the help of machine learning. 


![image](https://user-images.githubusercontent.com/66083579/153565249-2c0b4272-0286-40cc-b23b-c08c55821ae7.png)


With the help of Machine Learning, businesses can automate routine tasks. It also helps in automating and quickly create models for data analysis. Various industries depend on vast quantities of data to optimize their operations and make intelligent decisions. Machine Learning helps in creating models that can process and analyze large amounts of complex data to deliver accurate results. These models are precise and scalable and function with less turnaround time. By building such precise Machine Learning models, businesses can leverage profitable opportunities and avoid unknown risks.


Image recognition, text generation, and many other use-cases are finding applications in the real world. This is increasing the scope for machine learning experts to shine as a sought after professionals.  


# Some Important   Terminology of Machine Learning:

⦿    **Model**:   Also known as “hypothesis”, a machine learning model is the mathematical representation of a real-world process.
                   A machine learning algorithm along with the training data builds a machine learning model.


![image](https://user-images.githubusercontent.com/66083579/153604645-1beb79cc-8af3-4cb5-bb10-2fe07d07e4a7.png)



⦿    **Feature**:   A feature is a measurable property or parameter of the data-set.



![image](https://user-images.githubusercontent.com/66083579/153604937-4b95a4ca-7f64-4e83-adb5-833f0a6e3df1.png)




⦿    **Feature Vector**:   It is a set of multiple numeric features. We use it as an input to the machine learning model for training                             and prediction purposes.


![image](https://user-images.githubusercontent.com/66083579/153606298-bc4bcf08-0937-410e-b3be-1e817f10daad.png)




⦿    **Training**:     An algorithm takes a set of data known as “training data” as input. The learning algorithm finds patterns in                           the input data and trains the model for expected results (target). The output of the training process is the                           machine learning model.

![image](https://user-images.githubusercontent.com/66083579/153634887-c829d14c-5e56-4de2-8179-e869295ce48d.png)





⦿    **Prediction**:   Once the machine learning model is ready, it can be fed with input data to provide a predicted output.


![image](https://user-images.githubusercontent.com/66083579/153634679-36cc3634-ad63-469c-b1af-3b570e1268ee.png)


⦿    **Target(Label)**:   The value that the machine learning model has to predict is called the target or label.



⦿    **Overfitting**:   When a massive amount of data trains a machine learning model, it tends to learn from the noise and inaccurate data entries. Here the model fails to characterise the data correctly.


![image](https://user-images.githubusercontent.com/66083579/153635242-dd6e8015-ebec-4e76-8b7a-98f39068f950.png)



⦿    **Underfitting**:   It is the scenario when the model fails to decipher the underlying trend in the input data. It destroys the accuracy of the machine learning model. In simple terms, the model or the algorithm does not fit the data well enough.


![image](https://user-images.githubusercontent.com/66083579/153635688-9cc4b491-6a04-48ac-9176-43fe9f692ff4.png)


⦿    **Accuracy**:   Percentage of correct predictions made by the model.


![image](https://user-images.githubusercontent.com/66083579/153636863-3058e7cd-1854-45c0-bc81-a27a2fad336e.png)


⦿    **Algorithm**:   A method, function, or series of instructions used to generate a machine learning model. Examples include linear regression, decision trees, support vector machines, and neural networks.


![image](https://user-images.githubusercontent.com/66083579/153637336-2a7625ab-2f53-4b2f-a6be-f633640a10b9.png)


⦿   **Attribute**:   A quality describing an observation (e.g. color, size, weight). In Excel terms, these are column headers.


![image](https://user-images.githubusercontent.com/66083579/153637743-7dce9c90-2a6d-4890-b246-42b2f8217768.png)



⦿    **Bias metric**:  

![image](https://user-images.githubusercontent.com/66083579/153637980-8d654b52-f94b-47e4-9cd2-0769db36c0ee.png)



What is the average difference between your predictions and the correct value for that observation?

**•** **Low bias** could mean every prediction is correct. It could also mean half of your predictions are above their actual values and half are below, in equal proportion, resulting in low average difference.


**•**  **High bias** (with low variance) suggests your model may be underfitting and you’re using the wrong architecture for the job.


![image](https://user-images.githubusercontent.com/66083579/153638311-0b29849f-b967-444f-8eae-262624d832d4.png)



⦿    **Bias term**:   Allow models to represent patterns that do not pass through the origin. For example, if all my features were 0, would my output also be zero? Is it possible there is some base value upon which my features have an effect? Bias terms typically accompany weights and are attached to neurons or filters.   

![image](https://user-images.githubusercontent.com/66083579/153638829-33d54dbf-0302-4d80-ac9d-79e507d48a11.png)



⦿    **Categorical Variables**:   Variables with a discrete set of possible values. Can be ordinal (order matters) or nominal (order doesn’t matter).

![image](https://user-images.githubusercontent.com/66083579/153639107-ca3666ee-028e-4eaa-b016-6e28cacde744.png)



⦿    **Classification**:   Predicting a categorical output.

![image](https://user-images.githubusercontent.com/66083579/153632378-8cb799a5-6323-4441-b5e4-379968855cac.png)



**•**  **Binary classification** predicts one of two possible outcomes (e.g. is the email spam or not spam?)

![image](https://user-images.githubusercontent.com/66083579/153632619-0f9e335b-ea6c-48fd-add2-06d4e5f29fb7.png)



**•**  **Multi-class classification**  predicts one of multiple possible outcomes (e.g. is this a photo of a cat, dog, horse or human?)

![image](https://user-images.githubusercontent.com/66083579/153632985-df5da0a3-80a6-4404-ad44-00879a418281.png)



⦿    **Classification Threshold**:  The lowest probability value at which we’re comfortable asserting a positive classification. For example, if the predicted probability of being diabetic is > 50%, return True, otherwise return False.


![image](https://user-images.githubusercontent.com/66083579/153633277-08f63ac1-026f-476a-911e-8d7a335ecee2.png)


⦿    **Clustering**:  Unsupervised grouping of data into buckets.

![image](https://user-images.githubusercontent.com/66083579/153633406-e310974b-a156-4174-a6f4-6b2a73515f4e.png)



⦿    **Confusion Matrix**:  Table that describes the performance of a classification model by grouping predictions into 4 categories.

![image](https://user-images.githubusercontent.com/66083579/153633751-48d608be-e4c5-40af-b5b0-a6fbec4dc4f1.png)



**•**  **True Positives**: we correctly predicted they do have diabetes


**•**  **True Negatives**: we correctly predicted they don’t have diabetes

**•**  **False Positives**: we incorrectly predicted they do have diabetes (Type I error)

**•**  **False Negatives**: we incorrectly predicted they don’t have diabetes (Type II error)



![image](https://user-images.githubusercontent.com/66083579/153633892-b434a87e-acb4-412b-af5b-5d34e487d0b9.png)


⦿   **Continuous Variables**:  Variables with a range of possible values defined by a number scale (e.g. sales, lifespan).

![image](https://user-images.githubusercontent.com/66083579/153639766-540aa02f-d314-4e25-85a8-a88ec41aaf40.png)


⦿    **Convergence**:   A state reached during the training of a model when the loss changes very little between each iteration.


![image](https://user-images.githubusercontent.com/66083579/153649990-b7808c33-641a-4212-a455-c8d5530ddff5.png)



⦿    **Induction**:  A bottoms-up approach to answering questions or solving problems. A logic technique that goes from observations to theory. E.g. We keep observing X, so we infer that Y must be True.





⦿    **Deduction**:    A top-down approach to answering questions or solving problems. A logic technique that starts with a theory and tests that theory with observations to derive a conclusion. E.g. We suspect X, but we need to test our hypothesis before coming to any conclusions.



![image](https://user-images.githubusercontent.com/66083579/153650275-272a93a3-054c-4c93-9831-7c2ac6022b45.png)


⦿    **Deep Learning**:   Deep Learning is derived from one machine learning algorithm called perceptron or multi layer perceptron that gain more and more attention nowadays because of its success in different fields like, computer vision to signal processing and medical diagnosis to self-driving cars. As all other AI algorithms deep learning is from decades, but now today we have more and more data and cheap computing power that make this algorithm really powerful to achieve state of the art accuracy. In modern world this algorithm knowns as artificial neural network. deep learning is much more than traditional artificial neural network. But it was highly influenced by machine learning’s neural network and perceptron network.  


![image](https://user-images.githubusercontent.com/66083579/153650678-5301fb93-e8c7-4d1d-ac2c-451d769ec246.png)



⦿    **Dimension**:    Dimension for machine learning and data scientist is differ from physics, here Dimension of data means how much feature you have in you data ocean(data-set). e.g in case of object detection application, flatten image size and color channel(e.g 28*28*3) is a feature of the input set. In case of house price prediction (maybe) house size is the data-set so we call it 1 dimentional data.


![image](https://user-images.githubusercontent.com/66083579/153651654-0fe43fce-1c49-4856-af0d-0dafe05e60d3.png)

![image](https://user-images.githubusercontent.com/66083579/153651734-e86260d1-127e-4fe9-8f8d-264fcd15b5c6.png)


⦿     **Epoch**:     An epoch describes the number of times the algorithm sees the entire data set.


![image](https://user-images.githubusercontent.com/66083579/153652801-96f812d4-b235-4c9d-aea4-c78347fa8043.png)



⦿    **Extrapolation**:   Making predictions outside the range of a dataset. E.g. My dog barks, so all dogs must bark. In machine learning we often run into trouble when we extrapolate outside the range of our training data.


![image](https://user-images.githubusercontent.com/66083579/153653175-03d6cbbd-fdb2-490e-9d20-db9b8044223f.png)


⦿    **False Positive Rate**:Defined as
                            **FPR=1−Specificity=FalsePositives/FalsePositives+TrueNegatives**
                            
                           The False Positive Rate forms the x-axis of the ROC curve.
                           
  
![image](https://user-images.githubusercontent.com/66083579/153653589-93306c06-f4bb-40af-bfc0-828a6e0a17d6.png)



⦿    **Feature Selection**:    Feature selection is the process of selecting relevant features from a data-set for creating a Machine Learning model.


![image](https://user-images.githubusercontent.com/66083579/153653874-6e1cbe4c-118f-42d2-a258-45009ac51bd8.png)



⦿     **Gradient Accumulation**:  A mechanism to split the batch of samples—used for training a neural network—into several mini-batches of samples that will be run sequentially. This is used to enable using large batch sizes that require more GPU memory than available.


![image](https://user-images.githubusercontent.com/66083579/153654973-1d5f944f-a1e3-4973-8082-aa10651f9bf6.png)




⦿     **Hyperparameters**:  Hyperparameters are higher-level properties of a model such as how fast it can learn (learning rate) or complexity of a model. The depth of trees in a Decision Tree or number of hidden layers in a Neural Networks are examples of hyper parameters.


![image](https://user-images.githubusercontent.com/66083579/153655453-cadd13b3-0700-4bf3-bf31-f6a3388a0832.png)



⦿     **Instance**:   A data point, row, or sample in a dataset. Another term for observation.



⦿     **Label**:  The “answer” portion of an observation in supervised learning. For example, in a dataset used to classify flowers into different species, the features might include the petal length and petal width, while the label would be the flower’s species.


![image](https://user-images.githubusercontent.com/66083579/153656087-6a6c436a-5702-4925-ae46-6afe012c5aa6.png)


⦿    **Learning Rate**:  The size of the update steps to take during optimization loops like Gradient Descent. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.


![image](https://user-images.githubusercontent.com/66083579/153656356-7e5989bd-2fea-49c3-9ed1-3ae2eb9b2634.png)



⦿    **Loss**:   

**Loss = true_value(from data-set)- predicted value(from ML-model)**


The lower the loss, the better a model (unless the model has over-fitted to the training data). The loss is calculated on training and validation and its interpretation is how well the model is doing for these two sets. Unlike accuracy, loss is not a percentage. It is a summation of the errors made for each example in training or validation sets.


![image](https://user-images.githubusercontent.com/66083579/153717474-447ad67f-2277-4bd6-89f5-c03067abd418.png)



⦿     **Neural Networks**:   Neural Networks are mathematical algorithms modeled after the brain’s architecture, designed to recognize patterns and relationships in data.


![image](https://user-images.githubusercontent.com/66083579/153717572-dfd9607e-a4e6-435f-8d10-2cae81e01144.png)



⦿     **Normalization**:  Restriction of the values of weights in regression to avoid overfitting and improving computation speed.



![image](https://user-images.githubusercontent.com/66083579/153717668-2f7ebcd6-8d14-47d5-9505-41c7083be11e.png)




⦿     **Noise**:   Any irrelevant information or randomness in a dataset which obscures the underlying pattern.


![image](https://user-images.githubusercontent.com/66083579/153717737-b8f04bd5-edfe-44bc-b36f-72113aecb3fc.png)




⦿    **Null Accuracy**:   Baseline accuracy that can be achieved by always predicting the most frequent class (“B has the highest frequency, so lets guess B every time”).




⦿     **Outlier**:   An observation that deviates significantly from other observations in the dataset.



![image](https://user-images.githubusercontent.com/66083579/153717825-f74291ea-da08-4494-8a83-c8b9a0472a3d.png)



⦿      **Parameters**:   Parameters are properties of training data learned by training a machine learning model or classifier. They are adjusted using optimization algorithms and unique to each experiment.


**Examples of parameters include**:

**✦**   weights in an artificial neural network

**✦**   support vectors in a support vector machine

**✦**   coefficients in a linear or logistic regression

![image](https://user-images.githubusercontent.com/66083579/153722956-26bc2f9c-d227-4563-97ad-d1f084e9df7a.png)




⦿    **Precision**:   In the context of binary classification (Yes/No), precision measures the model’s performance at classifying positive observations (i.e. “Yes”). In other words, when a positive value is predicted, how often is the prediction correct? We could game this metric by only returning positive for the single observation we are most confident in.

**P =  TruePositives/TruePositives+FalsePositives**



⦿      **Recall**:   Also called sensitivity. In the context of binary classification (Yes/No), recall measures how “sensitive” the classifier is at detecting positive instances. In other words, for all the true observations in our sample, how many did we “catch.” We could game this metric by always classifying observations as positive.

 **R = TruePositives/TruePositives+FalseNegatives**
 
 
 ![image](https://user-images.githubusercontent.com/66083579/153723658-c9542d13-9328-45f8-a1c2-19e097a67ca0.png)



 ⦿    **Regression**:  Predicting a continuous output (e.g. price, sales).
 
 ![image](https://user-images.githubusercontent.com/66083579/153723756-631ccc35-04d6-4365-a2ea-41582c37d4bc.png)


 ⦿   **Regularization**:  Regularization is a technique utilized to combat the overfitting problem. This is achieved by adding a complexity term to the loss function that gives a bigger loss for more complex models.
 
 ![image](https://user-images.githubusercontent.com/66083579/153723830-cfba2e24-2e9f-4aae-a166-53bc59fd2bec.png)
 
 
 ⦿      **Reinforcement Learning**:   Training a model to maximize a reward via iterative trial and error.
 
 
 ![image](https://user-images.githubusercontent.com/66083579/153723881-52ba72b6-0d89-486f-9779-3e0bb4bd45af.png)


⦿  **ROC (Receiver Operating Characteristic) Curve**:  A plot of the true positive rate against the false positive rate at all classification thresholds. This is used to evaluate the performance of a classification model at different classification thresholds. The area under the ROC curve can be interpreted as the probability that the model correctly distinguishes between a randomly chosen positive observation (e.g. “spam”) and a randomly chosen negative observation (e.g. “not spam”).

 
 
 













