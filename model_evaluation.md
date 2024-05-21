[Return to Portfolio Page](https://sebastianghafafian.github.io/Portfolio/)


# Model Evaluation

Check out the project on [Github](https://github.com/SebastianGhafafian/CNN_Circle).

**Keywords**:
Convolutional Neural Networks (CNN), Pytorch, Regression, Data visualization, Pandas

## Introduction and motivation

In my master's thesis I wrote code to automatically detect ellipses in gray-scale images as a minor side project. Back then I had little knowledge of machine learning and I always wanted to revisit the topic later. After recently finishing my MicroMaster in Statistics and Data Science I thought I could try to solve a similar task using a neural network. I build a smaller working sample with a single circle in a image and fully connected feed forward networks. I tried to detect the (x,y) position of the circle center inside the image. That worked well. Then I introduced noise to see how far I could push the model and after that tried to detect the radius as well. In the end, I kind of drifted away from my inital motivation and became more curious how far I could push the model and whether a convolutional neural network could be a better solution for the task. The idea seemed interesting to me as I haven't seen to many applications of convolutional network for a regression task.

In the end, I ended up comparing two CNN models to predict the circle center and radius  and their performance for different noise levels. In this analysis, I explain the models I ended up using with a focus on the model's behavior on the test set.

## Dataset

40000 images have been created using the create_dataset.py. These binary images are 100 x 100 pixels containing a white circle with a random shape (radius r) and position (x,y). 
First the radius is randomly drawn from a discrete uniform distribution between 5 and 14 pixels. Then, the circle center position is randomly drawn from a discrete uniform distribution in x and y in the borders of [r,100-r] to avoid incomplete circles in the images. During training and testing, the data loader applies a transformation to the images adding Gaussian noise with zero mean and a varying standard deviation (from here on called noise level). The models are trained on different noise levels in range of [0.1, 0.3, 0.5, 0.7, 0.9, 1.1].
The image showcases the levels of noise for one image.
    
![png](model_evaluation_files/model_evaluation_1_0.png)
    


### Flaw in dataset

The randomly drawn radius of the circle determines the possible position of the circle center.
For example: If the drawn radius is 14, then a circle center is only possible in between [14,86] for x and y.
If the drawn radius is 5, then a circle center is possible in between [5,95] for x and y.
This will result in a smaller representation of circles with x or y positions near the edges between 5-15 and between 85-95. This flaw is later seen in the histograms. 
This should not be the biggest problem for convolutional neural networks
(maybe plot a 3d histogram or heat map of the x,y, positions)


## Model architecture and loss function

The loss function for this regression task has been set to the mean squared error. There are two models of varying complexity compared. The first convlolution network named **CustomCNN3** contains two convolutional layers for feature mapping and 3 fully connected layers for the regression task with a total amount of parameters of 23147.

<img src="./img/CustomCNN3.png" alt="CustomCNN3" width="800"/>


The second model called **CustonCNN5** contains 5 convolutional layers and three fully connected layers with a total amount of parameters of 270527.


<img src="./img/CustomCNN5.png" alt="CustomCNN5" width="800"/>

## Model training

The models are trained using 80 % of the data set for training. For each model for a specific level of noise, a gaussian filter is used to introduce noise to the picture. Each model is trained on the same split of training and test data totalling 12 different models. For most models it was sufficient to train for 15 epochs regarding the training time. Analysis during training has shown that the fluctuations in later learning stage come from the prediction of the radius, rather than the position. The following plot shows the test loss that occured during training. These plots are further discussed in the Lessons Learnt section of this report.


    
![png](model_evaluation_files/model_evaluation_6_0.png)
    


## Model Analysis


The model evaluation takes place on multiple levels. On qualitative level, a random sample is drawn and labels and predicted values are compared visually. This allows to gain some intuition of what might be going wrong or not as intended. This was mainly done during training of the model. 



On a quantitative level, the histograms of true labels and and predicted labels are compared to the performance of the model and to identify weaknesses. Further, the root mean squared error for each predicted variable, representing the average error in pixels for the estimation.

Is it more difficult for the model to predict labels for circles being at the edge of the picture?
How many weights had to be trained vs the amount of available data?

### Qualitative Analysis

To get an overview of the model's performance, it is useful to showcase the predicted values on the actual images.
    
![png](model_evaluation_files/model_evaluation_9_1.png)
    

### Quantitative Analysis

### Distribution Analysis of x,y and r

Plotting the histgrams of the predictions vs. the labels enable to evaluate the model performance per estimated variable. To minimize the amount of diagrams, the data is divided in low, mid and high noise with the values 0.3,0.7,1.1 respecively. The histograms of x and y are very similar, so only x and r are diplayed. 

    
![png](model_evaluation_files/model_evaluation_12_1.png)
    

The distribution of the predicted and the true x positions are very similar across the models which shows that the position is very well estimated even for high noise for which the circles are barely visible to the human eye. The histogram also show the minor flaw of how the data set was generated. The bins under 15 and over 85 contain far less observations that fade out to zero on both sides of the spectrum. The initial fear that a underrepresentation of those x positions lead to a worse predicition did not seem to have materialized.


![png](model_evaluation_files/model_evaluation_14_1.png)
    


The distribution of the predicted and the true r positions are very similar across the models and different levels of noises. It is quite impressive that there are is a relatively small amount of predicted values outside the range of true labels. For high noise, the models' predictions seem to fade out at both sides of the spectrum. 

### MSE for position and shape

The MSE is analyzed for the test set for the positional variables x,y and the shape variable r independantly to further dive into the models behavior. 

    
![png](model_evaluation_files/model_evaluation_18_0.png)
    


Both models show a very similar level of noise for x and y, which is to be expected.
The less complex model CNN3 shows a linearly increasing RMSE for increasing noise. This trend is not seen in the CNN5 model, yet the overall levels of RMSE are lower for the CNN3 model.
For very high noise of 1.1 (which makes the cirlces barely visible), the CNN3 starts to incur higher errors for x and y. The CNN5 model only makes a average error of estimation x and y under 2 pixels, which is quite impressive.

During training some models reacted very differently than others. CNN3 with a noise level of 0.9 reacted very differently in training and requires a lower learning rate, as it does not converge well. While observing the predictions, the position remained very robust, i.e. the histograms of x for predicition and true label did not change drastically. The fluctuations in the RMSE per epoch only seemed to occur in the optimization of the radius. This shows that parameters might affect the radius estimation more and might require a smaller learning rate than for x and y.

## Summary

Two convolutional networks have been built and compared to solve the task of finding the position and radius of a circle in noisy images. Considering the differing complexity of the model, both models performed similarly well for this regression task. Only for the highest level of noise, the more complex model CNN5 performed slightly better.



## Outlook

### Number of epochs and learning rate
After alot of investigations for the right combination of learning rate and batch size, I started training all models with the same learning rate of 1e-3 and batch size of 100 to keep as many parameters constant between the models.
After finishing all the training (which took some time) I realized that every noise level basically represents a slightly different task for the model to solve. Therefore, the optimal learning rate might also change. For some models, the test error did not converge well even if the number of epochs is drastically increased. These analyses are not included in this report. I investigated the MSE for each variable independantly during training and then continued training. The problem is the radius which has a harder time converging for both models. A very slight decrease in the learning rate might have solved this problem.
I would really like to come back to this project and visit the issue of convergence and whether the different scale of x and y compared to the radius play a role in this. 
Maybe a split model (presented in this [publication](https://www.researchgate.net/publication/355760763_Using_deep_learning_to_predict_the_East_Asian_summer_monsoon
)) for radius and position could stabilze the training. This could treat radius and position independantly, which could open up the potential for a model with less depth (less parameters) and therefore shorter training time.


Feel free to reach out to me or to check out my portfolio. I am happy to discuss my work and gain new insights. 


[Return to Portfolio Page](https://sebastianghafafian.github.io/Portfolio/)

