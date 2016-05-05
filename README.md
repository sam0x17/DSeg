# Delvr
## Detection and Localization via Image Retrieval

Delvr is a novel end-to-end pipeline for object detection, and then
3D localization / pose estimation on rigid objects for which high quality
man-made 3D models are readily available. The Delvr pipeline can be broken
up into the following steps.

## Training Pipeline

1. Use a high quality 3D model of the desired object class to generate tens
   of thousands of translucent photo-realistic images of that object at various scales,
   lighting settings, and orientations. These images will be used to train
   the pipeline to perform 3D pose estimation for that object class.

2. Overlay the translucent object images onto random scene images from the SUN
   scene images dataset (must be careful not to use scene images that already
   contain instances of our training object) to simulate realistic background
   noise. Use each object image multiple times.

3. Randomly erase regions of some of the object images to simulate partial
   occlusion (optional).

2. Run the Simple Linear Iterative Clustering (SLIC) algorithm on each of
   the scene images, resulting in superpixels. Use gSLICr so this only takes
   5-10ms at most.

4. Train a feed-forward neural network to take in a vectorized representation
   of a superpixel and the top 5 nearest superpixels (measured by centroid) as
   input, and output a single value indicating the network's confidence that
   the middle superpixel is part of an instance of our object.
   * The middle superpixel should be represented solely by its mean HSL color.
   * Adjacent superpixels should be represented by their mean HSL color and their euclidean
     distance from the middle superpixel.
   * These values should be arranged in a way that is topologically consistent
     with the original image.
   * We can take advantage of the fact that we already know the ground truth
     segmentations of our object instances (if any) in each image to train the
    network in a supervised fashion.

5. Train a Kohonen Self Organizing Map (SOM) to take in superpixel representations of
   the original object, using the same representation as before, except the entire object.
   * The object images should all be re-scaled to the same dimensions and have SLIC re-run.
   * Set the number of nodes to equal the number of unique poses in the database.
   * Do not terminate training until at least 98% are of the poses are mapped uniquely.
