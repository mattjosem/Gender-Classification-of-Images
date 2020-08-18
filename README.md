#### This is a project utilizing a convolutional neural network and machine learning techniques/libraries to classify profile pictures based on the gender. The accuracy is higher when the image is of a single individual of decent quality.

#### How to run: 

The command will be (using terminal):

	python3 image_gender_classifier.py -i <input directory> -o <output directory> 
	
> The input file must be a reachable directory with images to be tested. The output file must be an existing directory which the output xml files will be output. Each XML contains > the image’s predicted gender and the image’s file name without the file extension (you will be able to see which photo was classified via the file name).*

#### lbpcascade-frontalface.xml: 
This is strictly used with OpenCV in the scripts that require facial recognition to be used on images. 

#### image_model.py: 
This script is used to train a model on given input images and classify based on gender. Uses facial detection and convolutional neural network to classify the images’s gender based on the image. Images are loaded in, turned grayscale, facial detection used, and made uniform with identical pixel width and pixel height. Then, a model is created with a convolutional neural network as the framework to train the model on the input images. This is not the main script to be run. This simply trains the model.

*NOTE: The file structure of the training data was very specific, image’s file names were the ‘user-id’, so retraining the model would require identical directory/file structure or an alteration to the script.*

#### saved_image_model.h5: 
This is the saved model, generated when image_model.py is run. This is done so that every time images are being tested a new model is not trained every time. Having to retrain a model every instance images are to be tested increases computation time dramatically. 

#### image_gender_classifier.py: 
This is the main image-testing script. Within, images are turned grayscale, facial detection is used to find strictly the faces within, and made uniform with the same pixel width and height. This is then put in the model to predict the gender of the image. Output is to the output directory from the command
