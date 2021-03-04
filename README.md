# Allometry Data

OCR images of old allometry data.

Trying to OCR the images directly has not worked well. The pages are old and never were of high quality to begin with.  We're going to break this into three steps.
1. Clean up the images before the OCR step.
    1. Remove stray marks and fix problems with fonts and bad printing, etc.
    1. We're going to train a neural net to do this, either a denoising autoencoder or a U-net.
    1. We will generate and save formatted text for the pages then we can generate an image for the text as ground truth and then dirty the image. We can then use these images to train the net.
1. OCR the images.
    1. The plan is to use tesseract to do this.
1. Clean up the text after the OCR.
    1. Compare the OCR output to the text generated in step 1.
    1. If they compare favorably then just use the OCR output.
    1. If they do not, then we will need to do our best to correct the problems. This may be as easy as simple text manipulation, or it may require another neural network (probably a transformer network).
