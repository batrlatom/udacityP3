# Behavioral Cloning Project

Third project of the Udacity self-driving cars nanodegee is to teach neural network predict steering angle of the wheel from camera images.
We are provided with car simulator and our task is to train network to drive simulated car around the test track. 
---
To complete the project, we have to meet several goals:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/drivenet.png "Model Visualization"
[image2]: ./examples/stats.png "Train set histogram"
[image3]: ./examples/video.gif "Run video"



# 1. Included files
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing a trained convolution neural network 
* readme.md summarizing the results

# 2. Running the code
First, we need to download and install Udacity [simulator](https://github.com/udacity/self-driving-car-sim), run it, select track and choose autonomous mode.
Then, using the Udacity provided simulator and sourced drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 image_output_dir
```
# 3. Model Architecture
My model implements NVIDIA [drivenet](https://arxiv.org/pdf/1604.07316.pdf) in Keras. Architecture consist of four convolutional layers, followed by dropout and four dense layers, as seen on the image below ![drivenet][image1]

The model includes ELU layers to introduce nonlinearity. Directly in the model, data is cropped using Keras 2Dcropping layer and resized and normalized using a lambda layer with included TF backend.  The model used an adam optimizer, so the learning rate was not tuned manually.



# 4. Data aquisition

I aquired data from both tracks, using ps4 controller in the following way:

* 3 laps in the normal direction
* 2 laps driving opposite direction
* 1 lap recovery driving from the side of the road to the center. 

So to capture good driving behavior, I first recorded three laps on track one using center lane driving and two laps using the same technique, but driving in the opposite direction. Same was done for the second track. Next, I tried to get data which cover recovering from the side of the road back to the center. I created few dozen of such data samples for each track.

There were about 52 000 data points, each consisted from steering angle and tree camera images ( left, right, center ). I split data into training and validation sets in the 80% / 20% ration directly with the pandas lib ( file utils.py : pandas_split(X, y)) . Data was shuffled by creating masking array and splitting data into two distinct set according the mask.


![stats][image2]


You can see, that data distribution is little uneven, it will be fine to aquire more data with the steering to the right.


# 5. Overfitting reduction
The model contains dropout layer in order to reduce overfitting. I used 50% dropout prob rate. Second technique to prevent overfitting was to augment sourced data. Data was randomly rotated and translated. Model was trained on datasets over both tracks.



# 6. Training phase
Training took about 5 epochs and after that period, I achieved about 4% validation loss rate. So I tried to drive around the track. First attempt was distaster and car drove into the watter. I tried to get more data including driving around the watter and second attempt worked much better. I tried to use my model on second track, but I was unable to pass the first curve.


# 7. Final run 
Final run can bee seen in the video.mp4 file. It is obvious that model is able to learn, but would be much better to have more data. I was not able to go through the second track, but I think this is due lack of the data with steep steering angles. When I get hands over new GPU, I will try to update the model with more data. 
![run][image3]



