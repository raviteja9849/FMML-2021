# FMML[Foundations in Modern Machine Learning]-2021

![image](https://user-images.githubusercontent.com/66083579/153562815-2d778c69-dcdf-492d-a5e5-8cf259883029.png)

# What is Machine Learning?

Machine Learning is an Application of Artificial Intelligence (AI) it gives devices the ability to learn from their experiences and improve their self without doing any coding. 

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


![image](https://user-images.githubusercontent.com/66083579/153987843-f6761179-c834-4325-afe7-decf9f3a520b.png)



⦿    **Feature Selection**:    Feature selection is the process of selecting relevant features from a data-set for creating a Machine Learning model.



![image](https://user-images.githubusercontent.com/66083579/153604937-4b95a4ca-7f64-4e83-adb5-833f0a6e3df1.png)





![image](https://user-images.githubusercontent.com/66083579/153653874-6e1cbe4c-118f-42d2-a258-45009ac51bd8.png)








⦿    **Feature Vector**:   It is a set of multiple numeric features. We use it as an input to the machine learning model for training                             and prediction purposes.


![image](https://user-images.githubusercontent.com/66083579/153606298-bc4bcf08-0937-410e-b3be-1e817f10daad.png)




⦿    **Training**:     An algorithm takes a set of data known as “training data” as input. The learning algorithm finds patterns in                           the input data and trains the model for expected results (target). The output of the training process is the                           machine learning model.

![image](https://user-images.githubusercontent.com/66083579/153634887-c829d14c-5e56-4de2-8179-e869295ce48d.png)


⦿     **Testing**:     In machine learning, model testing is referred to as the process where the performance of a fully trained model is evaluated on a testing set. ... There are a number of statistical metrics that can be used to assess testing results including mean squared errors and receiver operating characteristics curves.


![image](https://user-images.githubusercontent.com/66083579/153986790-c9ddc334-358c-49ac-8fc7-73a5e75b2988.png)



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


![image](https://user-images.githubusercontent.com/66083579/153988766-2126263e-5248-4b68-867f-52d3157d3693.png)





![image](https://user-images.githubusercontent.com/66083579/153988983-63d42080-f4f9-4e1f-9cd6-458d5570cd6f.png)



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



⦿     **Discrete Variables**:  A Discrete variable can take only a specific value amongst the set of all possible values or in other words, if you don’t keep counting that value, then it is a discrete variable aka categorized variable.

Example: Number of students in a university.



![image](https://user-images.githubusercontent.com/66083579/153990323-82773fc0-cfde-4d1b-8ee2-cc20ad589f02.png)



⦿    **Convergence**:   A state reached during the training of a model when the loss changes very little between each iteration.


![image](https://user-images.githubusercontent.com/66083579/153649990-b7808c33-641a-4212-a455-c8d5530ddff5.png)





⦿    **Divergence**:   A  state during the training of a model when the loss changes is significant or unpredictable at each iteration and  it doesn't convergence at any no of iterations.



![image](https://user-images.githubusercontent.com/66083579/153991325-ae54ff56-f92f-467b-93d3-a604e675b0df.png)



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




⦿     **Interpolation**:   Interpolation is an estimation of a value within two known values in a sequence of values. Polynomial interpolation is a method of estimating values between known data points.



![image](https://user-images.githubusercontent.com/66083579/153653175-03d6cbbd-fdb2-490e-9d20-db9b8044223f.png)






⦿    **False Positive Rate**:Defined as
                            **FPR=1−Specificity=FalsePositives/FalsePositives+TrueNegatives**
                            
                           The False Positive Rate forms the x-axis of the ROC curve.
                           
  
![image](https://user-images.githubusercontent.com/66083579/153653589-93306c06-f4bb-40af-bfc0-828a6e0a17d6.png)






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





![image](https://user-images.githubusercontent.com/66083579/153993570-06915849-ebaa-4ced-b753-887b7daddc00.png)


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

 
 
 
 ![image](https://user-images.githubusercontent.com/66083579/153994679-c7ff963d-f2d4-4f1f-8fdb-0e2b6ef5a2ac.png)




![image](https://user-images.githubusercontent.com/66083579/153994700-5ea7da74-1569-4883-bff0-a65577ead489.png)



![image](https://user-images.githubusercontent.com/66083579/153994724-c640fd0b-e292-46da-8aab-b69fa08bffb0.png)



#   ROC  CURVE

![image](https://user-images.githubusercontent.com/66083579/153994801-7773d451-4a08-4d2e-9a10-b44c9ec386c0.png)




 ![image](https://user-images.githubusercontent.com/66083579/153994826-c7a3ee70-cb85-42bb-8362-f341c1efd5cb.png)

 
 
 ![image](https://user-images.githubusercontent.com/66083579/153995592-28a495b8-a5f5-45e9-9a6e-4e91d45659b5.png)



**Segmentation**:  It is the process of partitioning a data set into multiple distinct sets. This separation is done such that the members of the same set are similar to each otherand different from the members of other sets.



![image](https://user-images.githubusercontent.com/66083579/153995639-09c19515-e3f5-4127-833e-5a0b7cb2cbc9.png)



**Specificity**:   In the context of binary classification (Yes/No), specificity measures the model’s performance at classifying negative observations (i.e. “No”). In other words, when the correct label is negative, how often is the prediction correct? We could game this metric if we predict everything as negative.

  **S=TrueNegatives/TrueNegatives+FalsePositives**



![image](https://user-images.githubusercontent.com/66083579/153996257-dd9f88b6-263d-4d66-8b6e-03716e26c3cb.png)



**Supervised Learning**:  Training a model using a labeled dataset.


![image](https://user-images.githubusercontent.com/66083579/153996536-35d7585a-e473-4f3c-92cd-aba3942e4671.png)



**UnSupervised Learning**:   As the name suggests, unsupervised learning is a machine learning technique in which models are not supervised using training dataset. Instead, models itself find the hidden patterns and insights from the given data. It can be compared to learning which takes place in the human brain while learning new things. It can be defined as:


**❝Unsupervised learning is a type of machine learning in which models are trained using unlabeled dataset and are allowed to act on that data without any supervision.❞**


Unsupervised learning cannot be directly applied to a regression or classification problem because unlike supervised learning, we have the input data but no corresponding output data. The goal of unsupervised learning is to **find the underlying structure of dataset, group that data according to similarities, and represent that dataset in a compressed format.**


![image](https://user-images.githubusercontent.com/66083579/153997247-8bcce4c7-adcf-46c6-84fd-19dbff5c26d1.png)



**Transfer Learning**:  A machine learning method where a model developed for a task is reused as the starting point for a model on a second task. In transfer learning, we take the pre-trained weights of an already trained model (one that has been trained on millions of images belonging to 1000’s of classes, on several high power GPU’s for several days) and use these already learned features to predict new classes.


![image](https://user-images.githubusercontent.com/66083579/153997692-90cca64c-c2d1-47f9-b2e9-97e8a613d898.png)



**Type 1 Error**:  False Positives. Consider a company optimizing hiring practices to reduce false positives in job offers. A type 1 error occurs when candidate seems good and they hire him, but he is actually bad.



**Type 2 Error**:
False Negatives. The candidate was great but the company passed on him.




**Universal Approximation Theorem**:  A neural network with one hidden layer can approximate any continuous function but only for inputs in a specific range. If you train a network on inputs between -2 and 2, then it will work well for inputs in the same range, but you can’t expect it to generalize to other inputs without retraining the model or adding more hidden neurons.



![image](https://user-images.githubusercontent.com/66083579/153998385-9f332608-ab93-41cc-8ea8-a2f6b50cd3d4.png)



**Validation Set**: A set of observations used during model training to provide feedback on how well the current parameters generalize beyond the training set. If training error decreases but validation error increases, your model is likely overfitting and you should pause training.


![image](https://user-images.githubusercontent.com/66083579/153998852-26ba02d8-e31f-44d0-82c8-5267bb2ce1a2.png)




**Variance**:

How tightly packed are your predictions for a particular observation relative to each other?

Low variance suggests your model is internally consistent, with predictions varying little from each other after every iteration.
High variance (with low bias) suggests your model may be overfitting and reading too deeply into the noise found in every training set.




# TOOLS  USED FOR MACHINE LEARNING :

![image](https://user-images.githubusercontent.com/66083579/154214651-2b87fc21-71d8-4417-87c4-bc5bb896776b.png)





![image](https://user-images.githubusercontent.com/66083579/154214486-55d3d211-002f-4b4e-a479-594e7b15ca52.png)








![image](https://user-images.githubusercontent.com/66083579/154214746-81c7b9f0-af87-42aa-b8f2-00f53207740f.png)




#  APPLICATIONS OF  MACHINE LEARNING   IN REAL WORLD:


# 1.Autonomous Cars and Navigation:

![image](https://user-images.githubusercontent.com/66083579/154221790-05f1a1d2-3352-4f98-b207-dc4bdb221501.png)



![image](https://user-images.githubusercontent.com/66083579/154216712-d1b7067b-7190-444d-bcae-7b27715deffc.png)


**Object Detection with LIDAR**

![image](https://user-images.githubusercontent.com/66083579/154216730-67974553-031c-4148-93cb-e983ded6a3ba.png)



![image](https://user-images.githubusercontent.com/66083579/154216875-d92da9bd-27bf-4f3c-b1a3-e5c74ee6f9d3.png)



# **2.Personal  Assistants**:

![image](https://user-images.githubusercontent.com/66083579/154217648-7623a2eb-b5d6-4ddd-8844-99897be7ff37.png)



![image](https://user-images.githubusercontent.com/66083579/154217674-bea0d414-b90b-4a5b-9e5a-5f4a7c12ce28.png)



# **3.Create  Photographs and paintings**:


![image](https://user-images.githubusercontent.com/66083579/154218299-fc17b2e2-9f76-4489-a8d5-021b6ddfeb70.png)



![image](https://user-images.githubusercontent.com/66083579/154218406-fcd25de2-77b9-45cb-9dad-c3d61453c445.png)



![image](https://user-images.githubusercontent.com/66083579/154218715-df028691-2192-4cee-8a9a-f00fe4ee2eaf.png)




![image](https://user-images.githubusercontent.com/66083579/154218882-fbfc86c8-642d-4717-9ea6-d2d228a6605c.png)




![image](https://user-images.githubusercontent.com/66083579/154218960-87cccbb5-95d2-4a7b-9187-d13e391c93bd.png)


# **4.Chess/go Champions**:


![image](https://user-images.githubusercontent.com/66083579/154220900-f8ce81cc-4d31-4ccc-a7d5-65289acfb9d7.png)



![image](https://user-images.githubusercontent.com/66083579/154221102-01bd723c-8831-4326-8c34-b4ac57a76042.png)


![image](https://user-images.githubusercontent.com/66083579/154221526-5d300cfc-926e-4abb-ba89-83cbc7fcaafb.png)



![image](https://user-images.githubusercontent.com/66083579/154221558-2e0e60cc-7857-4dd9-bd18-2cff1379df05.png)



# 5.ML USED IN MEDICAL:

![image](https://user-images.githubusercontent.com/66083579/154222357-5bc94dd5-bcb3-4aec-894c-d3305a665f92.png)


# 6.ML USED IN SPACE IMAGING:


![image](https://user-images.githubusercontent.com/66083579/154226462-36d28d2b-9cbc-41a6-8eff-173bac29e718.png)



![image](https://user-images.githubusercontent.com/66083579/154226496-e826b7dd-a69c-40da-a2f6-6c3ef932e80a.png)



![image](https://user-images.githubusercontent.com/66083579/154226879-b498280f-d996-4c67-9ca3-73e62101e528.png)




#  7.AUTOMATED INSPECTION USING ML:

![image](https://user-images.githubusercontent.com/66083579/154227926-4ca148aa-91a2-47e2-a019-3a3e93732eab.png)


![image](https://user-images.githubusercontent.com/66083579/154227450-836993bd-4953-4381-a18e-e4ed73d89329.png)


![image](https://user-images.githubusercontent.com/66083579/154227879-609ab2ca-3d4f-475b-b60a-527d8c66a604.png)



![image](https://user-images.githubusercontent.com/66083579/154228039-6b6205d3-0939-44c0-87bc-aabfcb84c654.png)


![image](https://user-images.githubusercontent.com/66083579/154228093-2fa4f4e5-c74b-4248-8e79-4d9588f089c5.png)


# 8.BIOMETRICS: 



![image](https://user-images.githubusercontent.com/66083579/154232887-e33e87e1-b6c9-4749-9a18-0101c0d08d0c.png)


![image](https://user-images.githubusercontent.com/66083579/154232941-c1172936-da5b-46fa-8bda-75d64bff0a45.png)



![image](https://user-images.githubusercontent.com/66083579/154232977-c63ae0aa-6cd0-4ce7-8865-eb40869dfc76.png)



# 9.BROADCASTING:

![image](https://user-images.githubusercontent.com/66083579/154233088-fbe36d89-f624-42fb-9375-a6efb8944be5.png)




# 10.SURVEILLANCE:

![image](https://user-images.githubusercontent.com/66083579/154233713-334a821c-7ad4-4c76-bce9-a9557b963d36.png)


![image](https://user-images.githubusercontent.com/66083579/154233739-3cab3031-b185-49c3-b4df-e4c4a5fcff82.png)



![image](https://user-images.githubusercontent.com/66083579/154233786-6c1ac3ad-5486-4d43-ac3c-15d4e1db65bd.png)


# 11.AUTOMATED ASSEMBLY:

![image](https://user-images.githubusercontent.com/66083579/154234209-60ed352d-5069-43d3-8efc-24fe0ae992bc.png)


# 12.MAIL SORTING:
  
  
  ![image](https://user-images.githubusercontent.com/66083579/154234749-fe2a9121-339c-4ac6-b998-e87f76dd1808.png)



  
![image](https://user-images.githubusercontent.com/66083579/154234710-5514d771-171e-4014-9692-db6d46b61305.png)







