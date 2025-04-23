= NOTES REGARDING THE _"improved scnn model with LI layer"_
== #emph(text(red)[1st Problem :]) #highlight(fill:gradient.linear(silver, aqua))[SOLVED] 
_*$1^o$ We get random accuracy which upon printing the test images from batches we see that all predicted labels are 1.*_ 
\

- Tried to scale the encoded values x*10 inside the forward function BUT the values flactuate between 0.01 and 0.3 at best, might have to upscale this further. After training now we get all 0 labels.

- Upscaled by 100 so x*100 and now we print the final_ouytput and the mem_record after some prints the values actually got scaled down.

- #emph(text(red)[But we figured out that we have no spikes ! the spk_out tensor is zero.])

- Upon further investigating changing the x value doesn't change the 0 spike problem.

- Tried to add +10 to the x value in the encode inside the forward method.

- Removed the Normalize transform in the transfrom variable.

- Re added the Normalize transform and changed the learning rate perhaps it is stuck in a local minimum. And we also scale the x value by 50. all labels are 1.

- Checking for input spikes. WE HAVE A LOT at x*50 but at just x we get nada.

- #emph(text(green)[Changed the learning rate back to 0.005,and value x*10 and the model now converges casue the input spikes are ideal and we reach _*100%*_]) #highlight(fill:gradient.linear(silver, aqua))[SOLVED]

== #emph(text(red)[2nd Problem :]) #highlight(fill:gradient.linear(silver, aqua))[SOLVED]
_*$2^o$ for 50 time steps*_ 
- #emph(text(green)[Just switched the time steps from _300_ to _50_ and also changed the learning rate to _0.008_]) #highlight(fill:gradient.linear(silver, aqua))[SOLVED]

== #emph(text(red)[3rd Problem :]) 
_*$3^o$ le plots to be made are \
a) Input spikes\
b) Output Spikes after LIF \ 
c) Membrance traces after the final LI layer *_ 
- Plots
