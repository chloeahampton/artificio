"""

 image_retrieval.py  (author: Anson Wong / git: ankonzoid)

 We perform image retrieval using transfer learning on a pre-trained
 VGG image classifier. We plot the k=5 most similar images to our
 query images, as well as the t-SNE visualizations.

"""
import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from artificio.src.utils import makeDir
from artificio.src.CV_IO_utils import read_imgs_dir
from artificio.src.CV_transform_utils import apply_transformer
from artificio.src.CV_transform_utils import resize_img, normalize_img
from artificio.src.CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions
from artificio.src.autoencoder import AutoEncoder
from scipy.misc import imsave
import argparse

def execute_image_retrieval(args):
    # Run mode
    # modelName = "simpleAE"  # try: "simpleAE", "convAE", "vgg19"
    # trainModel = True
    modelName = args.model_name
    trainModel = args.train_model

    # Make paths
    # dataTrainPath = '/Users/chloehampton/ship_detection/auto_encoder_data/train_data/v1_1/'
    # dataTestPath = '/Users/chloehampton/ship_detection/auto_encoder_data/test_data/v1_1/'
    # outPath = '/Users/chloehampton/ship_detection/auto_encoder_data/CAI_results/v1_2/'
    dataTrainPath = args.data_train_path
    dataTestPath = args.data_test_path
    outPath = args.data_out_path
    print(dataTrainPath)

    # Read images
    extensions = [".jpg", ".jpeg"]
    print("Reading train images from '{}'...".format(dataTrainPath))
    imgs_train = read_imgs_dir(dataTrainPath, extensions, parallel=True)
    print("Reading test images from '{}'...".format(dataTestPath))
    imgs_test = read_imgs_dir(dataTestPath, extensions, parallel=True)
    shape_img = imgs_train[0].shape
    print("Image shape = {}".format(shape_img))

    # Build models
    if modelName in ["simpleAE", "convAE"]:

        # Set up autoencoder
        info = {
            "shape_img": shape_img,
            "autoencoderFile": os.path.join(outPath, "{}_autoecoder.h5".format(modelName)),
            "encoderFile": os.path.join(outPath, "{}_encoder.h5".format(modelName)),
            "decoderFile": os.path.join(outPath, "{}_decoder.h5".format(modelName)),
        }
        model = AutoEncoder(modelName, info)
        model.set_arch()

        if modelName == "simpleAE":
            shape_img_resize = shape_img
            input_shape_model = (model.encoder.input.shape[1],)
            output_shape_model = (model.encoder.output.shape[1],)
            n_epochs = 100

        elif modelName == "convAE":
            shape_img_resize = shape_img
            input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
            output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
            n_epochs = 500
        else:
            raise Exception("Invalid modelName!")

    elif modelName in ["vgg19"]:

        # Load pre-trained VGG19 model + higher level layers
        print("Loading VGG19 pre-trained model...")
        model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                            input_shape=shape_img)
        model.summary()

        shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
        input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
        n_epochs = None

    else:
        raise Exception("Invalid modelName!")

    # Print some model info
    print("input_shape_model = {}".format(input_shape_model))
    print("output_shape_model = {}".format(output_shape_model))

    # Apply transformations to all images
    class ImageTransformer(object):

        def __init__(self, shape_resize):
            self.shape_resize = shape_resize

        def __call__(self, img):
            img_transformed = resize_img(img, self.shape_resize)
            img_transformed = normalize_img(img_transformed)
            return img_transformed

    transformer = ImageTransformer(shape_img_resize)
    print("Applying image transformer to training images...")
    imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=True)
    print("Applying image transformer to test images...")
    imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=True)

    # Convert images to numpy array
    X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
    X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
    print(" -> X_train.shape = {}".format(X_train.shape))
    print(" -> X_test.shape = {}".format(X_test.shape))

    # Train (if necessary)
    if modelName in ["simpleAE", "convAE"]:
        if trainModel:
            model.compile(loss="binary_crossentropy", optimizer="adam")
            model.fit(X_train, n_epochs=n_epochs, batch_size=256)
            model.save_models()
        else:
            model.load_models(loss="binary_crossentropy", optimizer="adam")

    # Create embeddings using model
    print("Inferencing embeddings using pre-trained model...")
    E_train = np.array(model.predict(X_train))
    E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
    E_test = np.array(model.predict(X_test))
    E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
    print(" -> E_train.shape = {}".format(E_train.shape))
    print(" -> E_test.shape = {}".format(E_test.shape))
    print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
    print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))

    # Make reconstruction visualizations
    if modelName in ["simpleAE", "convAE"]:
        print("Visualizing database image reconstructions...")
        imgs_train_reconstruct = model.decoder.predict(E_train)
        if modelName == "simpleAE":
            imgs_train_reconstruct = imgs_train_reconstruct.reshape((-1,) + shape_img_resize)
        # plot_reconstructions(imgs_train, imgs_train_reconstruct,
        #                      os.path.join(outPath, "{}_reconstruct.png".format(modelName)),
        #                      range_imgs=[0, 255],
        #                      range_imgs_reconstruct=[0, 1])

        # for i, emb_flatten in enumerate(E_test_flatten):
        #     for j, img in enumerate(imgs_train):
        #         fname = os.path.join(outPath, "image_train_{}_{}.png".format(i, j))
        #         imsave(fname, img)

            # for j, img in enumerate(imgs_train_reconstruct):
            #     fname = os.path.join(outPath, "image_train_reconstruct_{}_{}.png".format(i, j))
            #     imsave(fname, img)

    # Fit kNN model on training images
    print("Fitting k-nearest-neighbour model on training images...")
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(E_train_flatten)

    # Perform image retrieval on test images
    print("Performing image retrieval on test images...")
    for i, emb_flatten in enumerate(E_test_flatten):
        _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
        img_query = imgs_test[i] # query image
        imgs_retrieval = [imgs_train[idx] for idx in indices.flatten()] # retrieval images
        outFile = os.path.join(outPath, "{}_retrieval_{}.jpeg".format(modelName, i))
        # plot_query_retrieval(img_query, imgs_retrieval, outFile)
        for j, img in enumerate(imgs_retrieval):
            fname = os.path.join(outPath, "image_train_{}_{}.jpeg".format(i, j))
            imsave(fname, img)

    # Plot t-SNE visualization
    # print("Visualizing t-SNE on training images...")
    # outFile = os.path.join(outPath, "{}_tsne.png".format(modelName))
    # plot_tsne(E_train_flatten, imgs_train, outFile)

    if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train-path', type=str, help='input training directory')
    parser.add_argument('--data-test-path', type=str, help='input testing directory')
    parser.add_argument('--data-out-path', type=str, help='output directory')
    parser.add_argument('--train-model', type=bool, help='whether to train or load model', default=True)
    parser.add_argument('--model-name', type=str, help='model type. can be vgg19, simpleAE, or convAE', default='simpleAE')
    execute_image_retrieval(parser.parse_args())
