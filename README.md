# Neural-Texture-Synthesis
## Abstract
Texture plays an important role in computer vision, graphics, and image encoding. Texture analysis and synthesis is a popular topic in the field of computer vision. In our project, we implement a simple neural feature synthesis model, load the pre-trained weights, try to generate kaleidoscopic features, and further improve our methods with advanced mathematical tools. Specifically, we used the pre-trained VGG-19 model as the core, extracting features from selected layers and computing the Gram matrices. Using a white noise image as input, we also passed it through the CNN and computed the loss function on the corresponding layers. The weighted loss from each layer was used as the final loss function, and gradient descent was performed to eventually find a new image that matches the Gram matrices of the original texture. In addition, we also implemented the CNNMRF model, which is also based on VGG-19 but further leverages MRF and EM algorithms. We conducted extensive experiments and comparisons, and explored control methods and styles.

## Parameters

| Argument                  | Default Value                     | Type     | Description                                                                 |
|---------------------------|-----------------------------------|----------|-----------------------------------------------------------------------------|
| `--model`                 | `vgg19`                           | `str`    | The pre-trained model to use for texture synthesis.                         |
| `--pooling`               | `avg`                             | `str`    | The pooling method to apply (e.g., `avg` for average pooling).              |
| `--rescale`               | `True`                            | `bool`   | Whether to rescale the weights or not.                                      |
| `--lr`                    | `0.1`                             | `float`  | The learning rate for the optimization process.                             |
| `--image_path`            | `images/pebbles.jpg`              | `str`    | The path to the source image for texture synthesis.                         |
| `--output_filename`       | `output.jpg`                      | `str`    | The filename to save the synthesized image.                                 |
| `--epochs`                | `1000`                            | `int`    | The number of epochs to run the synthesis process.                          |
| `--layer_list`            | `['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']` | `list`  | A list of layers in the model to use for texture synthesis.                 |
| `--method`                | `gram`                            | `str`    | The method used for texture synthesis (options: `gram`, `cnnmrf`).          |
| `--h`                     | `0.5`                             | `float`  | A hyperparameter (`h lambdas`) used in the synthesis process.               |
| `--patch_size`            | `7`                               | `int`    | The size of the patches used in the synthesis process.                      |
| `--lambda_orientation`    | `0`                               | `float`  | A weight (`lambda`) for orientation-related loss.                           |
| `--lambda_occurrence`     | `0.05`                            | `float`  | A weight (`lambda`) for occurrence-related loss.                            |
| `--lambda_colorstyle`     | `0`                               | `float`  | A weight (`lambda`) for color style-related loss.                           |
| `--style_image`           | `images/pebbles.jpg`              | `str`    | The path to the style image used for texture synthesis.                     |
| `--target_orientation_file` | `''`                             | `str`    | The path to a file containing target orientation information (if applicable).|
| `--output_folder`         | `./outputs`                       | `str`    | The folder where the output files will be saved.                            |
## Usage

To run the texture synthesis tool, use the following command:

```bash
python Synthesis.py --image_path path/to/your/image.jpg --style_image path/to/your/style.jpg
