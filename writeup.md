# Behavioral Cloning

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior

* Build, a convolution neural network in Keras that predicts steering angles from images

* Train and validate the model with a training and validation set

* Test that the model successfully drives around track one without leaving the road

* Summarize the results with a written report


[//]: # (Image References)

[model]: ./pics/model.png "Model Visualization"
[image2]: ./pics/center_lane_driving.png "Center Lane Driving"
[image3]: ./pics/recovering.png "Recovery Image"
[image4]: ./pics/flipped.png "Flipped Image"
[image5]: ./pics/loss.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` or `writeup_report.pdf` summarizing the results

####2. 2.	Submission includes functional code

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (`model.py` lines 166-220). It is similar to the NVIDIA model recommended in the project description. The shape of the images is 160x320x3, hence the input shape of the model is (160, 320, 3), (code line 148). The upper 60 rows and lower 20 rows of the images are removed (code line 153), as they provide no relevant information for the prediction of the steering angle. Afterwards the images are resized to 80x240x3, in order to have the same x-y ratio as in the NVIDIA example and to reduce the number of parameters. The information loss should be negligible. Next, the images are normalized using a Keras lambda layer (code line 164). After these preprocessing steps, the images are fed to the CNN.

The model includes ELU layers to introduce nonlinearity (code line 170).

####2. Attempts to reduce overfitting in the model

I also introduced dropout layers (`model.py` lines 171+172) as well as L2 regularization (code line 167) to the model in order to reduce overfitting. However, the performance was better without them. Therefore, in the final model I used early stopping to reduce overfitting and stopped the training after 3 epochs.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 144+145). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 222).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. In addition to the sample data provided by Udacity I used a combination of center lane driving (counterclockwise as well as clockwise) and some extra data for the curves. For recovering from the left and right sides of the road I used the images from the cameras on the left and right and adjusted the steering angle by +/- 0.2.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to

My first step was to use a convolution neural network model similar to the one from the NVIDIA paper. I thought this model might be appropriate for the simulator task because the authors of the paper reported good predictions of the steering angle based on real world images. Also, the general architecture of the CNN is state-of-the-art and proofed to be good for predictions based on images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set as well as on the validation set. But after 5 epochs the mean squared error on the validation set increased again. This implied that the model was overfitting.

To combat the overfitting, I reduced the number of training epochs. As mentioned above I also tried dropout and L2 regularization, but the results were worse.

Then I tried different parameter setting for the model (number of neurons, feature depths, batch size, weights initialization, batch normalization, conversion to YUV, masking region of interest, tuning the adjustment angle for cameras on the left and right side) and different training data (center lane driving, reverse driving, vehicle recovering from the left side and right side of the road back to center, extra data for difficult parts of the tracks, flipping the image)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in the curves. To improve the driving behavior in these cases, I collected better training data (which was not so easy at the beginning) and used the images from the cameras on the sides. It turned out, that with an appropriate model I needed just a little bit more training data apart from the sample data provided by Udacity to accomplish the first track.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road. Track 2 was more challenging. After fine-tuning the model and using also training data from track 2 I was able to drive almost one lap of track 2. But in one curve near the end the predictor is misled by the shadow of the mountain and the lane below, so the car keeps crashing into the poles. I tried to prevent it by equalizing the images, conversion to YUV and masking out the upper corners of the images. But nothing would help.

####2. Final Model Architecture

The final model architecture (model.py lines 151-220) consisted of a convolution neural network with the following layers and layer sizes:

| Layer | Description |
| ----- | ----------- |
| Input	| 160x320x3 RGB image|
| Cropping2D	| cropping upper 60 and lower 20 rows |
| Resize	| rescale image to 80x240x3|
| Normalization	| centering around zero with small standard deviation |
| Convolution 5x5	| 2x2 stride, valid padding, depth 24 |
| ELU	| worked better than ReLU |
| Convolution 5x5	| 2x2 stride, valid padding, depth 36 |
| ELU	| worked better than ReLU |
| Convolution 5x5	| 2x2 stride, valid padding, depth 48 |
| ELU	| worked better than ReLU |
| Convolution 3x3	| 1x1 stride, valid padding, depth 48 |
| ELU	| worked better than ReLU |
| Convolution 3x3	| 1x1 stride, valid padding, depth 64 |
| ELU	| worked better than ReLU |
| Flatten	| flatten convolution |
| Fully connected	| 100 neurons |
| ELU	| worked better than ReLU |
| Fully connected	| 50 neurons |
| ELU	| worked better than ReLU |
| Fully connected	| 10 neurons |
| ELU	| worked better than ReLU |
| Fully connected	| 1 neurons |

Here is a visualization of the architecture:

![model][model]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image2]

I then recorded the vehicle recovering from the left side and right side of the road back to center so that the vehicle would learn to return to center lane driving when it gets too near to the edges. These images show what a recovery looks like starting from the left side:

![Recovery Image][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help the model to better generalize. For example, here is an image that has then been flipped:

![Flipped][image4]

After the collection process, I had 26628 number of data points. I then preprocessed this data by flipping the images randomly and augmented the data by using the left and right images with an angle adjustment of +/-0.2. The rest of the image pre-processing is done in the Keras model as described above.
I finally randomly shuffled the data set and put 20% of the data into a validation set (code line 142).
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the loss function (mean squared error) – see the following figure:

![Loss][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
