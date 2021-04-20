# Allometry Data

OCR images of old allometry data.

Trying to OCR the images directly has not worked well. The pages are old and never were of high quality to begin with.

1. Train some models to recognize the distorted and broken characters. (MNIST style)
1. Dissect images into lines and then into characters on each line.
1. For every model, read the allometry sheets character by character and run each character through the model.
1. Create a single "best" version by using the models as an ensemble.
1. Cleanup text. See if we can differentiate 0s from Os and 1s from Is etc. Also use spell checking and other heuristics.
1. Write the data to CSV files. Try to recreate that tables as closely as possible. For other reports formats separate the labels and values.

## Experiments

The [notebooks](scratch) directory contain experiments where I figure out how to approach things. The code is not expected to be useful outside that.

## Failed Approaches

This code is in the [history](history) directory, so you don't have to go Git spelunking to find them.

### Failed approach #1
We tried running these through the tesseract OCR program in various configurations. It's a great program, but it's not designed to work with "distressed" fonts.

### Failed approach #2
We're going to break this into three steps.
1. Clean up the images before the OCR step.
    1. Remove stray marks and fix problems with fonts and bad printing, etc.
    1. We're going to train a neural net to do this, either a denoising autoencoder or a U-net.
    1. We will generate and save formatted text for the pages then we can generate an image for the text as ground truth and then dirty the image. We can then use these images to train the net.
1. OCR the images.
    1. The plan is to use tesseract to do this.
1. Clean up the text after the OCR.
    1. Compare the OCR output to the text generated in step 1.
    1. If they compare favorably then just use the OCR output.
    1. If they do not, then we will need to do our best to correct the problems.
   
No matter how much the images were cleaned they just would not work with the major open source OCR programs.
