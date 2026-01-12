Resume
GAGAN NAGARAJ
Boulder, CO | 602.388.5158 |gagan.konana@gmail.com

LinkedIn: linkedin.com/in/gagan-konana
GitHub: github.com/gagankonana
Portfolio: gagankonana.github.io


TECHNICAL SKILLS
Programming/ Scripting : C++, Python, Go.		Database : PostgreSQL, SQLite3, Neo4j, MongoDB. 
OS			 : Linux, FreeRTOS.		Tools	   : Git, Jenkins, Argo CD, AWS, Datadog, Vault, Yocto.
Frameworks	 	 : GTest, FastAPI, Django, Selenium, Pytest, Celery, RabbitMQ, Tensorflow.

OPEN SOURCE CONTRIBUTION
[C++] Fixed memory access violation issue in Tensorflow’s tf.raw_ops.ResourceSparseApplyAdagrad* APIs by adding input rank validation and raising downstream Python error.
[C++] Refactor packet delay to remove static initial delay and add support for dynamic session start based on audio stream latency accumulation, allowing more control on jitter handling and support planned improvements on RTCP.
[C++] Fixed and added test for open issue to safely throw error when setting end dates for tasks in ‘pending‘ state.

EXPERIENCE
Splunk | Software Engineer | Boulder, CO, USA	Dec 2024 - Present
Created thread-safe TTL enabled data structure on LRI cache in C++, for time-bound caching with O(1) peek, which is used to limit concurrent LDAP binds to prevent Denial of Service attack (through thread exhaustion).
Implemented nested OAuth 2.1 Authorization Code + PKCE flow (per RFC 7636), building Resource Server and Authorization Server in C++ and Go, to allow integration with MCP servers and all Cisco platforms.
Built Splunk Tokens, RFC 7519-complaint JWT with custom claims, to unify auth framework within spunk ecosystem.
Supported Identity infra for product expansion into GCP & 4 AWS regions in 3 weeks, aim to generate $600M by FY27.
Automated steps to access AlloyDB on GCP using auth proxy through debug pod, reducing DB access time by 90%.
Created on-call onboarding document based on my onboarding learnings and automated ~15% of manual steps.

AdviNOW Medical | Software Engineer | Scottsdale, AZ, USA	Jan 2024 - Nov 2024
Designed Metaprogramming-based scoring algorithm to handle 100+ questionnaire templates, and integrated with athenaOne APIs for submitting patient answers and scores into EMR, saving ~4 min of front desk time per patient.
Led EMR integration refactor, built using Django, PostgreSQL, and Celery, transitioning from data-dependent triggers to RabbitMQ event-based task triggers for full task control and to support upcoming features on product roadmap.
Re-architected Rules Engine (FastAPI + SQLAlchemy service) to support multiple language translations on scale.
Increase convergence speed by 15% with minimal change in accuracy of Bayesian theorem based Diagnostic Engine.

AdviNOW Medical | Software Engineer Intern | Scottsdale, AZ, USA	Jun 2023 - Dec 2023
Built microservice using FastAPI to generate History of Present Illness with mako template for Doctor App and EMR.
Optimized set of microservices by implementing multithreading, minimizing PostgreSQL queries and redefining cache structure from Neo4j, resulting in a performance boost of over 55% for each endpoint.

Agile-Displays (Store Intelligence) | Software Engineer Intern | Pleasanton, CA, USA	Apr 2022 - Aug 2022
Implemented workflow priority queuing algorithm on IoT device timeslot to optimize thread allocation by 30% in C++.
Modeled lexical analyzer for Domain Specific Language support to enable custom flows for QA team to experiment.
Integrated SQLite3 with C++ to move device specific cloud data onto edge devices, reducing boot latency by 10%.
Debugged Memory fragmentation of C++ application using Valgrind and Bash scripts (to track RSS and VSZ).
Refactored Cortex-A53 - M4 binary communication protocol, cutting 75% Radio Frequency cycle wastage.
Built a Test application for Windows in C# to replicate Vantiq cloud application to automate regression testing.

eSamudaay | Software Engineer Intern | Bengaluru, Karnataka, India	Oct 2021 - Feb 2022
Constructed APIs for a decentralized commerce platform using Django with integration of PostgreSQL.
Architected and Collaborated with frontend team to build a backend feature leveraging item labels to allow users to build dynamic custom UI for their marketplace.
Demonstrated problem-solving skills by identifying and fixing bugs through corner case unit testing in over 15 existing endpoints.
Integrated additional unit tests for existing APIs using Pytest framework, leading to a improvement of 8% in overall coverage.

Graphene AI | Software Engineer Intern | Bengaluru, Karnataka, India	May 2021 - Sep 2021
Achieved a high level of accuracy in language detection and filtering by remodeling and deploying a Deep Learning model using Tensorflow with an F1-score of 0.93 to pass reviews into "Sentiment Analysis" stage of pipeline.
Improvised Logistic regression part of the core Sentiment Analysis model to increase the overall accuracy by 2%
Optimized data extraction processes by automating it with web scraping frameworks (Scrapy and Selenium).
Improved data manipulation with development of data visualization tools using Python and Streamlit.

EDUCATION
Master’s, Computer Science | Arizona State University | GPA: 3.93	2022- 2024
Bachelor's, Computer and Information Science | National Institute of Technology, Karnataka | GPA: 7.72	2018- 2022


Project reports are saved in project_reports folder
PROJECT
Similarity-Aware Channel pruning for Convolutional Neural Networks | Thesis | Report	Jul 2021 - Mar 2022
A channel pruning method using feature map similarity to compress CNNs for resource constrained deployment.



Projects:

Harmony | Dec 2023
Abstract—In today’s rapidly moving and frequently stressful world, prioritizing both physical and mental well-being is es- sential. Our application addresses these priorities by not only focusing on maintaining heart and mental health but also by pro- viding recommendations such as restaurant suggestions, weather- related advice like monitoring AQI levels, and even guidance on when to wear appropriate clothing like a jacket or carry an umbrella. Additionally, Harmony conveniently summarizes the daily calories burnt based on data collected from various smartwatches.
Index Terms—Guardian Angel, Harmony, Health Application, Health Companion
I. INTRODUCTION
In an era characterized by the swift pace of modern living and the prevalence of stress, the imperative of nurturing physi- cal and mental health has become paramount. Acknowledging this fundamental need, our project explores an innovative application designed to prioritize and bolster these essential facets of wellness. This paper delves into the multifaceted functionalities of our application, which not only focuses on preserving cardiovascular and mental well-being but also offers a comprehensive array of user-centric features.
Central to our application is its dedication to promoting a holistic approach to health, transcending the mere monitoring of vital signs. It extends its utility by furnishing users with invaluable recommendations, encompassing diverse aspects of daily life. From providing tailored restaurant suggestions to offering pertinent weather-related guidance, such as real-time monitoring of Air Quality Index (AQI) levels, or even offering advice on appropriate attire choices like wearing a jacket or carrying an umbrella, the application aims to seamlessly integrate health-conscious decisions into users’ routines.
Moreover, a distinguishing feature of this application, named Harmony, lies in its ability to synthesize and present crucial data about users’ physical activity. By harnessing the data acquired from various smartwatches, Harmony succinctly compiles and presents insights into the daily calories ex- pended, empowering users with a comprehensive overview of their fitness endeavors.
Through this project, we elucidate the novel features and functionalities of Harmony, demonstrating its potential to sig- nificantly enhance users’ well-being by amalgamating health management with practical, everyday decision-making.
II. ARCHITECTURE
The application can be segmented into four primary com- ponents, each of which corresponds to a distinct facet of its functionality. These components are prominently featured with dedicated buttons on the mobile app’s home page. Data ag- gregation occurs from diverse origins, including smartwatches, the Google Location API [1][2], and user questionnaires. This amalgamated data is transmitted to our AWS server, wherein a Flask App [3], scripted in Python and leveraging an array of libraries and APIs, processes the information. Subsequently, the outcomes are relayed back to the Mobile App for display. The operational workflow of each component is outlined below:


Comparative Study of Medical Data Recognition | Dec 2022
Abstract
This project’s goal is to determine which classifica- tion algorithm provides the best estimate for accu- rately predicting health problems for the following datasets: breast cancer detection, heart failure pre- diction, and stroke prediction. To process the data and optimize the output, we employ appropriate normalization and regularization techniques. As a result of the proposed project, a comparative anal- ysis of the algorithms will be performed.
1 Introduction and Motivation
This project focuses on implementing various clas- sifiers based on input features to predict or clas- sify a disease. In this project, we used three differ- ent health datasets and implemented three differ- ent classifiers. The primary goal of the project is to evaluate and compare different classifiers’ perfor- mance. The proper application of Machine Learn- ing in the health industry will have a positive im- pact by allowing early disease detection and treat- ment by doctors. As a result, performance analy- sis and comparison will provide insight into which classier can be used to solve similar problems. This is the driving force behind the proposed topic. The topic of the project is chosen based on its relevance to the course. The algorithms are chosen in ac- cordance with the course syllabus. Additionally, Normalization and Regularization are being im- plemented.
2 Problem Description
Our study shows that one of the most signifi- cant causes of death worldwide today is heart
disease. We also studied that due to unhealthy lifestyle of people, most of them in their middle age are more likely to suffer from strokes. It is linked to high mortality and morbidity rates. One of the most prevalent diseases affecting women in today’s world is breast cancer. In 2020 there were lakhs of deaths from breast cancer world- wide which is something to be worried about. One of the crucial step in rehabilitation and treatment is obtaining an accurate and prompt diagnosis. There might be more fixes or longer endurance rates assuming illness is recognized before. A significant reduction in disease-related mortality can be achieved through early diagnosis and treat- ment. Due to the digitization of this world, ma- chine learning plays an important role in the di- agnosis of these diseases. During the COVID-19 pandemic, we observed that many of us were con- fined to our homes, increasing the demand for dig- itized medical records. In this project, we use ma- chine learning models on which distinct methods of feature extraction and data preprocessing will be done. Furthermore, by analyzing the performance of each disease and predicting it more accurately, our project aims to identify the best algorithms for each disease so that it would help people to diag- nose the disease much earlier.
3 Dataset
All of the datasets were obtained from Kaggle and then spilt into train and test datasets before model- ing them.
3.1 Heart Failure
The heart failure dataset has a total of 917 data points and 11 features. In 918 data points, 507 had
December 8, 2022
1

 heart disease and 410 did not have heart disease.
 Figure 1: Heart Failure label distribution
3.2 Stroke
The stroke dataset has a total of 5110 data points with 10 features. In 5110 data points, 249 are suf- fering from stroke and 4861 do not have a stroke.
Figure 3: Breast Cancer label distribution
4 Methodology 4.1 EDA
Exploratory Data Analysis was done on each dataset to get insights into the feature-diagnosis co-relation.
4.1.1 Breast Cancer Data-set
Fig 4 maps the direct co-relation of features on the Breast Cancer diagnosis. Lighter-coloured map- pings have high co-relation.
The below graphs represent the individual fea- tures vs Breast cancer diagnosis.
Figure 4:
EDA conclusion for Breast cancer Dataset: Radius and density: higher radius and lower den- sity can mean a higher chance of breast cancer
  Figure 2: Stroke label distribution
3.3 Breast Cancer
The breast cancer dataset has total of 569 data points with 31 features. In 569 data points 212 are malignant and 357 are benign.
2

Texture means: higher texture means a higher chance of positive diagnosis.
Smoothness and Concavity have a higher influence on the diagnosis output.
4.1.2 Heart Failure Data-set
6 maps the direct co-relation of features on Stroke diagnosis. Lighter-colored mappings have high co-relation.
EDA conclusion for Stroke: From the Fig 6, sam- ples in the dataset with female gender is more than that of male, 26.5% of of samples have hyperten- sion. 11.6% of the samples were never married. 28.1% of the data formerly smoked, 16.9% still smoke and the remaining never smoked. 59.8% of the data are privately employed, 26% are self em- ployed, 13.3% have govt jobs and remaining are children. 54% of the data are urban population. In total, 18.9% of the samples have heart disease. The age of the data range from 55-80 with avg glucose level of 80-2000 and bmi 20-40.
4.2 Algorithms
I. k-NN:
Figure 7: K-Nearest Neighbors
Regression and classification both make use of the k-nearest neighbors (k-NN) method of supervised learning. k-NN attempts to pre- dict the appropriate class for the test data by calculating the distance between the test data and all of the training points[1]. The k number of points that are closest to the test data will then be chosen. The k training data class with the highest probability is selected after the k- NN algorithm examines the likelihood of test data belonging to that class. Figure 1 shows the working of k-NN.
II. SVM: A support vector machine (SVM) is a supervised ML model that is used as a classifi- cation algorithm for group classification prob- lems. Each observation is plotted as a point in n-dimensional space(n is the number of fea- tures in the given dataset)[2]. An optimal hy- perplane is built for classifying the data points
  Figure 5: Features Analysis of Heart Failure Dataset
Fig 12 maps the direct co-relation of features on Heart Disease diagnosis. Lighter-colored map- pings have high co-relation.
Age, Fasting, and Oldpeak features have the high- est influence on the heart disease dataset.
4.1.3 Stroke Data-set
 Figure 6: Features Analysis of Stroke Dataset
3

 into their appropriate classes. Figure 8 shows the working of SVM where A and B are sup- port vectors and C is the hyperplane.
 Figure 8: Support Vector Machine
III. Logistic Regression: A classification algo- rithm from Machine Learning called logistic regression is used to predict the likelihood of particular classes based on some dependent variables. In a nutshell, the logistic regression model computes the logistic of the result by adding up the input features (most of the time with a bias term).Reference image is shown
Figure 10: Random Forest
V. XGBoost: Extreme Gradient Boosting (XG- Boost) is a distributed, scalable gradient- boosted decision tree (GBDT) machine learn- ing library. In this calculation, decision trees are made in sequential order. The weights of the variables play an important role in this algorithm. The weight of variables that are predicted wrong by the tree is increased and these variables are then put into the next tree that is being built and the iteration continu- ous. These individual classifiers/predictors then ensemble to the best model[3].
VI. Gaussian Naive Bayes: The Bayes algo- rithm is implemented using the Naive Bayes theorem[4]. Each feature pair is assumed to be conditionally independent. While working on continuous data it is assumed that each class’s value has a gaussian normal distribution. The formula is given as below,
 P(xi|y)=q
1 2πσy2
 (xi−μy)2 exp − 2
   Figure 9: Logistic regression
IV. Random Forest: It is a classification system that comprises many decision trees. A ”For- est” is created using many trees and out of an ensemble of decision trees, which are commonly trained using the ”bagging” ap- proach. The main notion behind the bagging approach is that combining several learning models improves the final outcome and per- formance of the model.
VII. Decision Tree: Decision Trees constantly sep- arate data based on a parameter.Data is di- vided into divisions and further divided on each branch using this iterative method. The categorization model is created with the help of a decision tree. The attribute is represented by each node in the tree, and the attribute’s potential value is represented by each branch that descends from that node.
4
2σy

4.3 Tuning and Evaluation of Models 4.3.1 Tuning techniques
• GridSearchCV: It is a method for determining the best values for a grid’s parameters from a given set. It is essentially a method of cross- validation. It is necessary to enter both the model and the parameters.In our project,the best hyperparameters of the model was found using the GridSearchCV.
• Smote: SMOTE is a data augmentation algo- rithm that creates synthetic data points from the original data points.As we encountered imbalanced dataset in the project which is done we used SMOTE to handle the problem.
• K-fold: The holdout method is repeated k times, with each of the k subsets serving as the test set and the other k-1 subsets serv- ing as the training set, in the K-Fold valida- tion technique. The project uses K-fold cross- validation for the resampling procedure to evaluate machine learning models as we had a limited data sample.
4.3.2 Evaluation Metrics
The evaluation metrics used in our project to eval- uate the best-built models are as follows. The met- rics we used are Accuracy, Precision, Recall, F-1 Score, and AUC.


Publication: Interactive System for Toddlers using Doodle Recognition | July 2021

Abstract
Typing using the keyboard or using a mouse is hard for small chil- dren. In this paper, we proposed an interactive system to improve the learning ability of a toddler. The proposed doodle recognition system provides an attractive and efficient way to interact toddlers with com- puter systems by following the Human-Computer Interaction guidelines and deep learning. The most common practice that toddlers develop is scribbling random images, so we decided to use this skill to provide a gateway for the toddlers to interact thus and learning with computers by using our proposed simple interface. Toddlers come across develop- mental milestones in their phase of life. Developmental milestones are milestones achieved by toddlers for instance saying their first word and putting their first step forward etc..Toddlers move about more in the second year and are more aware of themselves and their environment. Their drive to learn about new things and people is also growing. We have researched and studies show that visual learning is far more efficient when compared to listening and chanting the words taught to us. So in this paper, we present a way to bestow knowledge while children do what they are good at:(scribbling). When the toddler (user) starts to scribble or draw something on the screen, whiteboard, or paper; the application goes into input mode, and as soon as the drawing is stopped the image on the screen or whiteboard is processed by the trained CNN model and
1
2
Springer Nature 2021 LATEX template Interactive System for Toddlers using Doodle Recognition
the action is carried out based on the output of the model. The environ- ment is set up by the parents or day care workers, Users analysis was done on four questions and based on the age groups and background.
Keywords: Convolutional Neural Network (CNN), Doodle recognition, Human-Computer Interaction (HCI), Toddlers, Learning ability.
1 Introduction
Human-Computer Interaction (HCI) is a study that mainly focuses on human interaction with computers. Humans play the main role of a user in the study, and in this paper, we considered toddlers as users. HCI makes the usability and interaction with the computers for the user easier. The main objective of the HCI study is to make easy use of navigation through the technology for the users.
The early stages of life are very crucial for a successful human life. Jean Piaget [1], a child development theorist, found that in the first two years of life (Sensorimotor stage), children generally learn through their movement and their senses. When they cross the 12 months, there are two sub-stages in tod- dler cognitive (thinking) development. The first one, usually occurs between 12 –18 months. In this period, toddlers learn through the process of trial and error. They are continuously trying to implement ideas that ponder in their head. Communicating ideas with them is very vital. Doing so helps toddlers pro- cess the information they gather and see that one can respect their thoughts. The second stage is typically from 18 – 24 months and is the “Beginnings of Thought” stage. Here, the toddlers learn symbolic thoughts and begin to pre- tend play. The concept that one thing can represent another occurs here. For instance, a block can be a telephone, or a paper plate could be a pie [1].
Our paper focuses on these two sub-stages, where a child can interact with the system through scribbles or doodle drawing on a whiteboard or draw- ing board (connected to the computer) and learn simultaneously. We followed many HCI design rules such as the Fitts’ law, Hicks’ law, and Don Norman’s principles. Further, we proposed a CNN-based deep learning model to make the interactive system more efficient by real-time recognition of doodles. Sketches, unlike real photographs, are highly abstracted because they lack the rich elements of genuine photographs, such as diverse colours, backgrounds, and environmental information. Doodles or sketches are still meaningful enough to encompass a sufficient amount of significance despite all of these shortages and being drawn with only a few strokes [2].
The most well-known image identification and classification technique is the Convolutional Neural Network (CNN), often known as a ConvNet. Convolution is a mathematical linear action between matrices that gives it its name. CNN is a type of neural network that uses convolution rather than standard matrix multiplication in at least one layer [3]. A CNN’s main building block is the

Springer Nature 2021 LATEX template
Interactive System for Toddlers using Doodle Recognition 3
convolutional layer. The parameters of the layer are a series of learnable filters (or kernels) [4].
Although existing works provide good outcomes with respect to doodle recognition, there is not much work in the platform, which solves toddlers’ interaction with computers. So in this paper, we discussed a potential solution for the problem by using the limited skill of doodling that toddlers possess. To the best of our knowledge, the key contributions of this work are:
• Providing a way for toddlers to interact with the system and make sure their scribbling does not go wasted.
• Providing an alternative way of learning for young minds instead of traditional charts and books.
• Providing a platform for parents to keep track of the learning behavior of their kids.
In this new era of rapid development, product design must address not only practical and functional needs, but also emotional and psychological design needs, and in this proposed model, it must address the needs of a toddler[5]. The design and implementation of an effective approach to educating toddlers through doodles are novel in this study. Our major goal is to create a plat- form for toddlers to use their drawings and doodles to engage and learn with computers.
The remaining portions of the paper are structured in the following manner: The Human-Computer Interaction principles are discussed in Section 2, which provides an overview of how the suggested model adheres to these concepts. In Section 3, we address a background investigation as well as some related works that inspired the work that is being presented. The many phases that are involved in the proposed interactive system are broken down and explained in Section 4. In Section 5, additional information is provided regarding the tests that were conducted and the results that were achieved, as well as an analysis of the user experience. In the final part of the paper, that is, Section 6, we shall conclude the work and discuss some potential future paths.



Similarity-aware Channel Pruning | Dec 2021
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning system that can take an input picture, assign relevance (learnable weights and biases) to var- ious objects in the image, and distinguish between them. When compared to other classification methods, the amount of preprocessing required by a ConvNet is signif- icantly less. While basic approaches need hand-engineering of filters, ConvNets can learn these characteristics with enough training. Deep convolutional neural networks are di cult to deploy on resource-constrained devices because of their high compu- tation and memory requirements. We can tackle this obstacle by compressing the network by pruning it.
Figure 1.1.1: Pruning example.
Neural network pruning is a method of compression that involves removing weights from a trained model. Pruning a neural network may be done in a variety of ways. Weights can be pruned. Setting individual parameters to zero and making the network sparse accomplishes this. The number of parameters in the model would be reduced while the architecture remained same. You may also take out whole nodes
 1
from the network. This would reduce the size of the network design while maintaining the accuracy of the bigger network.
There are di↵erent types of pruning, for example Weight-level pruning, Channel Pruning, Filter-level pruning etc
Weight-level pruning - This method try to find out unimportant weights and set them to zeros
Filter-level pruning - This method locate and remove unimportant filters in con- volutional layers
Channel Pruning is the process of eliminating weights from the network that connect neurons from two neighbouring levels. When a DL model contains a large number of convolutional layers, the process of obtaining a near-optimal solution with a stated and acceptable reduction in accuracy becomes more complex. After performing channel pruning, the outputs of the next layer should be reconstructed by combining the corresponding channel-wise weights of all filters of the next layer.That means when two feature maps are identified strong relative similarity, one of them can be removed safely, the related channel-wise weights will be discarded too. By applying such an addition operation to the corresponding channel-wise weights, can get a reconstructed feature map that is not much di↵erent from the original feature map. This process is known as Weight Reconstruction
We implement a Similarity-aware Channel Pruning technique in this project.By utilising the similarity of a layer’s output feature maps, we aim to eliminate dupli- cated channels. Then, using the Weight Combination technique, we produce new parameters for the following layer in order to rebuild the outputs. Our project’s main points may be described as follows: Instead of concentrating on the filter saliency, we do channel pruning by focusing on the redundancy of feature maps directly.A Similarity-aware Channel Pruning technique will be utilised to minimise repetition by rejecting the related channels of the next layer that use the deleted feature maps as inputs and deleting the unnecessary weight groups of the current layer that cre- ate similar outputs. We will use the Weight Combination approach to generate new parameters for the following layer, allowing us to reconstruct the outputs with fewer channels. Through fine-tuning, this technique makes it simpler to restore precision. Conduct experiments on CIFAR-10 to determine the e ciency Most of the existing
2

channel pruning approaches mainly prune unimportant filters by the filter saliency. However, filters with low saliency doesn’t always lead to low saliency to the outputs. Under such circumstance, the incorrect removal may result in the accuracy reduction of the model. Moreover, a direct method of channel pruning is to exploit the filter similarity and remove the redundant ones which have the similar values to others. However, such kind of methods have some limitations:
• Influenced by other factors (such as bias, batch normalization), similar filters don’t necessarily lead to similar feature maps. Therefore, the removal of the similar filters may result in the incorrect pruning.
• Di↵erent filters can also generate similar feature maps. Thus, merely comparing the similarity of the filters may lead to ignoring the removal of redundant filters
1.2 MOTIVATION
Our project is motivated by two factors. First, despite Convolutional Neural Net- works’ (CNNs) tremendous performance in di↵erent visual identification tasks, their high computational and storage costs prevent them from being used in resource- constrained devices. To get around this, we’ll utilise channel pruning to thin down the network. Second, current techniques perform channel pruning by reducing the feature map reconstruction error between the pretrained and pruned models. Al- though reconstructing feature maps can maintain the majority of the information in the learnt model, it has drawbacks.


Short Term Load Forecasting | Jan 2021
Information Technology Department, National Institute of Technology Karnataka, Surathkal
Atharv Belagali(181IT208) Gagandeep KN(181IT215) Prithvi Raj (181IT234)
Information technology Information Technology Information Technology
National Institute of Technology Karnataka National Institute of Technology Karnataka National Institute of Technology Karnataka
Surathkal, India Surathkal, India Surathkal, India
Abstract— We define Load forecasting as a method to predict the
future quantity of power or energy required to meet the supply.
In simpler terms Load forecasting is about estimating future
consumptions based on various data and information available as
per consumer behavior. Majorly it is used by power companies
and limit their resources consumption. There are various types of
load forecasting namely short-term which is applicable for a few
hours, medium-term for a few weeks up to a year and long-term,
whose time period is over a year. Short-term Load Forecasting
can help to estimate the load flows and to make decisions that
can prevent overloading. Timely implementations of such choices
result in improvement of network responsibleness and to reduced
occurrences of apparatus failures or blackout.
INTRODUCTION
In today’s deregulated economical world, we see that
sustainable living is the only way to go and for that we need to
know how much power is to be used and try not to waste
excess resources. This is where load forecasting comes into
picture. It has many applications including energy purchasing
and generation, load switching, contract evaluation, and
infrastructure development. We know that electrical energy
can’t be stored, it is whenever there is a demand for it.
Therefore, it is imperative that electrical appliances energy
intake load is foretold. This estimation is called load
forecasting. It is necessary for power system planning.
In this paper we will be focusing on Short term load
forecasting and some methods on how it can be implemented.
The period of short term load forecasting usually ranges from
an hour to a week. It gives us an approximation on the energy
usage and prevents overloading.
Let’s discuss the roles of Load forecasting:
⮚For proper planning of Power System.
• To forecast the future need for additional new generating
facilities.
• To determine the size of the plants.
• On correct values of demand, load forecasting will prevent
over designing of conductor sizes.
⮚ For proper planning of Transmission and Distribution
facilities.
•Wastage due to unplanning like the purchase of equipment
which is not immediately required can be avoided.
⮚ For proper Financing.
⮚ For proper Grid Formation.
The load on a Power Station never remains constant rather it
varies time to time, the variation of load with time is known as
load curve. They are usually measured or plotted per hour or
so on a daily, monthly, annual basis and hence the names:
•Daily load curves.
•Monthly load curves.
•Annual load curves.
With the Load curves we can determine: Variation of load
during different times, Total no. of units generated, Maximum
demand, Average load on a power station, Load Factor.
There are some factors that determine the results of load
forecasting namely:
•Weather conditions (Temperature and Humidity).
•Class of customers (residential, commercial, industrial,
agricultural, public, etc.).
•Special Events (public holidays, etc.).
•Electricity price.
The data we have used is taken from ENTSO-E Power
Statistics Data whose link can be found in the reference
section [5]
•We have collected corresponding weather data from NCEI
ISD [6]
•Factors applied : Time of the day, Day of the week ,
ENTSO-E Hourly Load, Weather condition.
Data processing is done as follows:
Firstly, a function is defined to convert a vector of time series
into a 2D matrix, next the dataset is split dataset: 90% for
training and 10% for testing, now the training set is shuffled.
(but do not shuffle the test set).
OBJECTIVES:
1.Determine the time and factors affecting short-term
forecasting
2.Use various models to do short term forecasting
3.Take outputs and compare the models and make inference
METHODOLOGY:
Network Training :
1.DBN:
It is generally referred to as a probability generation network
composed of several Restricted Boltzmann Machines (RBMs)
[13, 14]. The network basically is composed of three layers: a
visible layer, a hidden layer, and an output layer. So the
hidden and the visible layers are joined using weights, and all
the neuron each of them have their own offsets to represent
their weight. The output layer and therefore the previously
hidden layer form a BP neural network which is especially
wont to adjust the initial parameters of the hidden layer to
realize supervised training of the entire network. In the DBN
learning process, the data is input from the bottom layer and
then through the various hidden layers to complete the
training process. The learning process here is split up into two
sections: pretraining and fine tuning. The figure below shows
a DBN structure with n layers hidden.
2. K-means clustering:
The K-means algorithm is a typical clustering algorithm. The
basic notion of the algorithm is to split the samples into
different clusters by continuous iteration until the objective
function reaches its optimal value.
The specific procedure is as follows:
(1) Choosing K appropriate points as the initial clustering
centers., in this project, we have set k=365
(2) Calculate the value of d:
(3) Classify all the points of load according to their nearest
center points, thus dividing all the sample points into K
clusters.
(4) Calculate the middle of the mass of the K clusters and
update them as new agglomeration centers.
(5) Repeat the steps 2,3,4 and, keep iterating until the
clustering centers stop shifting.
Then while predicting we compare the input temperature and
other data points with the mass centers to find the cluster the
input points belong to produce the load output.
3.LSTM:
3.1. Training the LSTM Network
The LSTM network is meant to be told each the long and
short-run options of the coaching knowledge. Toward this
finish, the kind of input file has connexion to the effectiveness
of learning. If the info is provided that is leading toward
wrong direction or isn't enough to form the options clear, the
LSTM or RNN can learn consequently and can not predict or
forecast accurately. as an example, the yearly seasonality
within the provided knowledge is simply for one year. The
LSTM can take into account it a seamless trend and can
predict wrong and can lead toward zero values step by step.
However, if the info of quite one year is given, LSTM will
learn to predict the yearly seasonality too. Similarly, the
choice of input file additionally contributes when deciding the
accuracy of the network performance. The coaching method
knowledge that has all types of seasonality and trends except
the yearly seasonality.
• optimizer="rmsprop"
• No. of Training Iterations (Max Epochs) = 100
batch size=512
*activation = linear
model summary:
Layer (type) Output Shape Param #
============================================
=====================
lstm (LSTM) (None, None, 50) 10400
_______________________
dropout (Dropout) (None, None, 50) 0
_______________________
lstm_1 (LSTM) (None, 100) 60400
_______________________
dropout_1 (Dropout) (None, 100) 0
_______________________
dense (Dense) (None, 1) 101
============================================
=====================
Total params: 70,901
Trainable params: 70,901
3.2. Forecast
The trained LSTM network in the previous subsection is used
to forecast over the horizons of the given range or the
particular hour entered by the user.


Parallelizing a Chess Engine
Abstract—Computer AI is now ruling the 21st century and
the centuries yet to come. Since parallel computing is in such a
trend , it is natural that it complements heavy computation like
AI. In this paper we implement an application of AI using
parallel computing .We implement a parallel search algorithm
named root-splitting.The main idea of this technique is to ensure
that each node except for the root, is visited by only one
processor.
I. INTRODUCTION
Chess programming is regarded as one of the most
classical challenges ever encountered in AI. It may
seem easy at first glance but when we look profoundly,
there exist 10 to the power of 120 possibilities. Which
is a lot to comprehend and frankly difficult to
enumerate and execute them.
Fortunately for us, there exist an algorithm which
gives computers the ability to play chess, that is the
Minimax algorithm. The minimax; from a given node
in the search tree, computes all its children (that is the
set of all the possible moves from the current
configuration) and visits each child one by one,
computing each time an evaluation function that gives
away a kind of performance of that child. We repeat
that recursively until we reach the leaves. At the end,
the algorithm chooses the branch with the highest
evaluation. However, the search tree being too large,
we generally stop at a maximum depth, otherwise it
would be computationally heavy or worse: impossible.
By parallelizing the search, we can compute
independent branches simultaneously, gaining a
considerable time that we could use to visit an extra
layer.
Instead, we focused on optimizing it by parallelizing
the computation of independent branches. We also use
a bit of python script to display the difference between
the sequential and parallel search graphically.
II. OBJECTIVES
• To understand and implement a chess engine
sequentially and parallelly.
• To implement the chess game using a specific
parallel search algo.(Root-splitting).
• To compare and prove the fact using visual graphics
that parallel runs faster than sequential.
III. SYSTEM REQUIREMENTS
• Software requirements:
Windows/Linux /macOS
• Hardware Requirements:
i3 processor
4GB RAM
IV. LITERATURE SURVEY
1. Parallel Alpha-Beta Pruning of Game Decision Trees
;A Chess Implementation
- Kevin Steele:
An AI powered program that plays chess either with itself or
with an use. The most efficient method to traverse a
minimax game tree is the alpha-beta pruning algorithm. This
algorithm effectively prunes portions of the search tree
whose leaf node scores are known a priori to be inferior to
the scores of leaf nodes already visited. Unfortunately for
parallel computing enthusiasts, the alpha-beta algorithm and
its variations are inherently sequential. In fact, if great care
is not given in parallelizing the algorithm, parallel
performance can degrade well below the performance of the
serial algorithm.
2. Massively Parallel Chess - Christopher F. Joerg1 and
Bradley C. Kuszmaul:
Implementation of Chess game with different types of
algorithms. Uses a lot of algorithms: Negmax search
without pruning, Alpha-beta pruning etc.. And gives their
analysis. Since the algorithms are so many , the debugging
is going to be a problem and since parallel and sequential
almost use the same code.
V. METHODOLOGY:
The main idea of this technique is to ensure that each
node except for the root, is visited by only one
processor. To keep the effect of the alpha beta pruning,
we split the children nodes of the root into clusters,
each cluster being served by only one thread. Each
child is then processed as the root of a sub search tree
that will be visited in a sequential fashion, thus
respecting the alpha beta constraints. When a thread
finishes computing it’s subtree(s) and returns its
evaluation to the root node, that evaluation is
compared to the current best evaluation (that’s how
minimax works), and that best evaluation may be
updated. So to ensure coherent results, given that
multiple threads may be returning to the root node at
the same time, threads must do this part of the work in
mutual exclusion: meaning that comparing to and
updating the best evaluation of the root node must be
inside a critical section so that it’s execution remains
sequential. And that’s about everything about this
algorithm!
Given the time allowed for this project and the relative
simplicity of implementation, we came to choose this
algorithm for our project. We used the openmp library
to do multithreading for its simplicity and efficiency.
The full implementation of the root splitting algorithm
can be found on the repository. We'll just illustrate the
key (changed) parts of the original C code here.
We said that we parallelize at the root level, so for that
we parallelize the for loop that iterates over the
children of the root node for calling the minimax
function on them. for that, we use the following two
openmp directives:
#pragma omp parallel private (/*variables private to
thread here */) #pragma omp for schedule (dynamic)
The two above directives need to be put right before
the for loop to parallelize. In the first one we declared
the variables that must be private to each thread. In the
second directive we specified a dynamic scheduling
between threads, meaning that if a thread finishes his
assigned iterations before others do, he'll get assigned
some iterations from another thread, that way making
better use of the available threads.
Then we must ensure inside our for loop that after each
minimax call returns, the thread enters in a critical
section in mutual exclusion to compare (and modify)
the best evaluation of the root node. We do that using
the following directive:
#pragma omp critical { // Code here is executed in
mutual exclusion }
All the code written inside this directive will be run by
at most one thread at a time, ensuring thus the
coherence of the value of our best evaluation variable.
VI. RESULTS :
We display two graphs that compares sequential and
parallel parts : first graph plots every move on the
chess board and the other displays total time using a
bar graph respectively.
As the graph suggests the time taken by serial player
at each step is significantly more than parallel
player and also graph in Fig 6.2 and 6.5 Shows total
time taken by serial is much more compared to
parallel
Figure 5.1 Root splitting (dividing the nodes into threads(or clusters))
Fig 6.1 Graph showing the time taken by the player for every move till the end
Fig 6.2 Bar graph indicating total time taken by each player by the end of the game
Now we have a display of how the pawns in
the chess board are being played and in the end