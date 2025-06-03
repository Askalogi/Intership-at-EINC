#import "@preview/minimal-presentation:0.6.0": *
// #import "codly:1.3.0": *
// #show: codly-init.with()

// #set text(font: "Lato")
// #show math.equation: set text(font: "Lato Math")
// #show raw: set text(font: "Fira Code")

#show: project.with(
  title: "7 Weeks in EINC",
  sub-title: "Neuromorphic Computing",
  author: "Kalogirou Asterios",
  date: "28/04/2025",
  index-title: "Contents",
  logo: image("images/visions_logo_core_990000_noText_new.png"),
  logo-light: image("logoss.png"),
  cover: image("images/einc_thumb.jpg"),
  main-color: rgb("#811aff"),
  lang: "en",
)
= Beginning

== First Four Weeks 

- The first 4 weeks were spent on learning as much of the basics as possbile before getting some hands on experience so I dont get overwhelmed.
\
- A lot of reading into the team's projects and papers was performed, as well as some of the team's tutorials, which helped me to understand some of the modeling better and gave me a headstart regarding Neural Networks.
\
- Of course nothing beats hands-on experience so in those four weeks I also got accomodated with the main library that I was going to be using, *PyTorch*.
\
== First Four Weeks
- Some PyTorch tutorials were completed explaining the API and also training some models from scratch with the MNIST dataset.\
\
- This specific tutorial made me build a ANN and two CNN models to classify the MNIST and the Fashion - MNIST datasets.\
\
- With this new found knowledge it was time to move forward !
= Creating a dataset

== Dataset (Simple)
- After the first 2-3 weeks a project idea was being discussed and first task was officialy given.
  -  Starting goal:
Create a dataset with some parameterization, so it is _rather_ easy to classify.
After some iterations of this dataset and some consulting with my supervisor (E.A.) a final function following the sin function was chosen.
== Results for the Custom Dataset
#columns-content()[
  #figure(
    image("images/code1/code1.png",height: 12cm),
    caption : [Code for the main dataset function],
)<code1>][#text([Yes this is the pixel accurate test image, the size limitations were discussed with Elias and Eric (mostly between them) so it can later be implemented easier on hardware.])
  #figure(
    image("images/code1/code1_img1.png"),
    caption : [Pixel Accurate Image Result]
  ) <code1_im1>
]
== Results for the Custom Dataset
#columns-content()[
#figure(
  image("images/code1/code1_img2.png", height: 12cm),
  caption: [Scaled up plotted test Image],
) <code1_im2>
][#text([This is the plotted image that was randomly generated before.
\
A second python script was created to generate a *num_samples* images with randomized _orientation_, _stripes_ and _noise_. Here is the part of that code that customizes that:])
#figure(
image("images/code2/code2.png", height: 4cm, width: 14cm),

)<code2>
]
== Dataset (notes)
- The customizable features have a certain limitations that makes sense both _spatial-wise_ (e.g. the stripes cant be more that half the image size) and _logically-wise_ (e.g. after a certain noise threshold the images become _smooth_)
\
- Here the labels are encoded to *0* and *1* for horizontal and vertical lines respectively, also the images are saved in such a way : _image.index_encoded.label.png_ . \
- Meaning we can extract the label directly from the loaded image without the need of a ._csv_ file.

== Dataset initialization

- By importing the Dataset class from the module torch.utils.data we can create a custom class that inherits from the Dataset class of PyTorch, so we can create it to our liking.
- The vital methods are the transform and the root which enable the dataset to be tracked (located) and also transformed (from png files to Tensors for PyTorch usage)
#columns-content()[
  #figure(
    image("images/code3/code3.1.png", width: 100%),
    caption: [Import of Dataset],
  ) <code3.1>
][
  #figure(
    image("images/code3/code3.png", width: 90%),
    caption: [Custom class Dataset],
  ) <code3>
]


= Models
== Models

Many models were created until a desired model was finalized, the procedure that was recommended by my supervisor was to start broad with the limitations and slowly to start adding new important features.
\
This method was very effective regarding the learning and the results.
\

1. The first model was a simple SNN model that was tested using a version of the custom dataset that required a ._csv_ file.
  While the model did learn to differentiate vertical and horizontal lines, 30 epochs were neaded and it never fully converged for this simple task (reaching about 90% accuracy).


== Models 
2. In the second model we took the existing SNN model that was working and also added a convolution layer mainly using the norse library.

 
#figure(
  image("images/code4/code4.png", height: 8cm ), 
  caption: [1st SCNN model]
  )<code4>

== Models
This is a very inefficient model with many layers that are very unneccesary, to be exact, the model consists of two convolution layers, a pooling layer, 3 LIF layers and a fully connected (linear) layer.
\
The convergance happens extremely fast at aroun epoch 3 which means we can simplify this model quite a lot !
\
Since it works though, the next step was to add some kind of encoding or rather add the temporal dimension.
\
// The results that are as follows :\
// - Training:\
//       Epoch 1/10 - Loss: 64.8693 - Accuracy: 51.4286 %\
//       Epoch 2/10 - Loss: 37.6165 - Accuracy: 79.8571 %\
//       Epoch 3/10 - Loss: 13.7869 - Accuracy: 98.0000 %\
//       Epoch 4/10 - Loss: 10.4440 - Accuracy: 99.1429 %\
//       Epoch 5/10 - Loss: 9.9136 - Accuracy: 100.0000 %\

// The convergance happens really fast !

// - Testing : \
//   Test loss is : 0.0000 
//    Accurace is : 98.67 %
== Models
3. The third model's goal was to add the time dimension to the *mix*.
The results here were much harder to "make sense" because the model was now much simpler and a little harder for me to train since we added the time dimension.
\
// Not a lot of time was spend optimizing this model because the encoding we were striving for was a*_ #highlight("constant current over time method.", fill: rgb("#811aff8a"))_*
This was done by using a specific import :\

#underline(raw("from " + "norse.torch.functional.encode " + "import poisson_encode"))
\

Not a lot of time was spend optimizing this model because the encoding we were striving for was a*_ #highlight("constant current over time method.", fill: rgb("#811aff8a"))_*
And the new simpler and time encoded model looks something like this :
== Models
#columns-content()[
#figure(
  image("images/code5/code5.png", width: 14cm ), caption: "SCNN + TIME" 
)][
 - So here the inputs get encoded using the _poisson_encode_ function that was imported changing the shape of the tensors to [Time, Batch_Size,Channel, Height, Width] 

]
== Models 
4. Fourth model is almost identical to the previous one with the only difference being the encoding.
\
#underline(raw("from " + "norse.torch.functional.encode " + "import constant_current_lif_encode"))\
\
In order to rate this model, the class was decided with  highest spike count per sample.*_ #highlight("highest spike count per sample", fill: rgb("#811aff8a"))_* in the output layer.

Here an accuracy of $~$98 % was achieved not like the 100% convergance but still very good :
Here are some results taking random inputs to see if the model has a weakness to any specific pattern.
== Models
#columns-content()[
  #figure(
    image("images/code5/code5_im1.png", width: 14cm,
    ) ,caption: "Random Images in the Trained Model"
  )
][- Here we can see that the model struggles with a great number of stripes while also having high level of noise. 
\
- Still this is a very good result but we can make the model even better by changing the architecture and adding a LI layer. ]

== Models 
5. The fifth model is the final model that I have made completely from scratch (so not counting any tutorials) and its the best one to date.
The code for the model looks something like this : 
== Models
#columns-content()[
  #figure(
    image("images/code 6/code6.png"),
    caption: "Code block for the final model"
  )
][
- The network architecture includes a 
 - rate encoded input,
 - a convolutional layer with 2 filters size 4x4 ,
 - a LIF layer for the spikes , 
 - a max pooling layer which reduces the spatial dimensionality , 
 - a flatten + linear layer and finally 
 - a LI layer which models membrance voltage traces with decay, in order to capture long-term temporal information.]


== Models 
This model is both simple enough to move into hardware and also gets perfect results regarding the taks it is given.
\

The network fully converges at around epoch 9 meaning we get $~100%$ accuracy on our custom dataset and since is the best model so far some extra plots were made using a custom python function.

== Results of Final Model
#columns-content()[
  #figure(
    image("images/code 6/code6_im1.png")
  )
]
== Results of Final Model
#columns-content()[
  #figure(
    image("images/code 6/code6_im2.png"), caption: "Test Image"
  )
][
  #figure(
    image("images/code 6/code6_im3.png"), caption : "Input Spikes of the Test Image"
  )
]
== Results of Final Model
#columns-content()[
  #figure(
    image("images/code 6/code6_im4.png"), caption : "Hidden spikes over Time"
  )
][
  #figure(
    image(
      "images/code 6/code6_im5.png"
    ), caption: "Membrance Voltage Traces of the two Output Neurons over Time "
  )
]

== Comments 
- Here we can see that the output neurons are symmetric over the $y = 0 $ line meaning that classification is possible with only one output neuron.
\
- Having achieved our starting goal, an idea to change the dataset and make it a little harder was suggested.
\
- And so this exact model, with minor changes to the scaled encoded input values, was tested on a dataset that has the same parameterization as the previous one with the difference that now striped patches are randomly placed around a noisy grid.

== Comments 

#columns-content()[
  #figure(
    image("images/code 6/code6_im6.png"), caption : "Test image with an easier version"
  )
][- The dataset with the easier pictures got a training for $~18$ epochs and it fully converged at around 12 epochs
- While the dataset with the harder images is learning, more epochs are recquired and more changes on the model to make this model converge fully. ][
  #figure(
    image(
      "images/code 6/code6_im7.png"
    ), caption: "Test image with a harder version "
  )
]
== Comments 
#columns-content()[
  #figure(
    image("images/code7/code7_im1.png"), caption : "Test image of the easier version"
  )
][#figure(
    image("images/code7/code7_im2.png"), caption : "Input spikes over time")]
== Comments 
#columns-content()[
  #figure(
    image("images/code7/code7_im3.png"), caption : "Hidden Spikes"
  )
][#figure(
    image("images/code7/code7_im4.png"), caption : "Membrance Voltage traces in the 2 neurons")]

= Looking forward

== Next Steps 
- As of now the next direct step is going to be transfering this model and the (easier) dataset to hardware and to try and make it work there.(it has already begun)
- Also since the membrance voltage trace of the output neurons are symmetric the second existing output neuron could be changed, along with the way that the image is encoded through time to classify the rotation of it (clockwise - counterclockwise).
- Finally some other ideas might pop up which are always fun to talk about and implement.


= Conclusion

== Conclusion
- Over the course of these 7 weeks, I’ve had the opportunity to immerse myself deeply into the field of programming and neuromorphic computing . I have gained a solid foundation in PyTorch, built and trained spiking neural network models, and developed custom datasets—all of which helped me better understand both the theory and practical application of neural-inspired computation.

- Looking ahead, I’m enthusiastic about continuing this work—particularly as we begin transitioning models to hardware—and I’m eager to keep learning, improving, and contributing to the team's goals.

