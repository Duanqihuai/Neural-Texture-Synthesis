# Neural-Texture-Synthesis
## Abstract
Texture plays an important role in computer vision, graphics, and image encoding. Texture analysis and synthesis is a popular topic in the field of computer vision. In our project, we implement a simple neural feature synthesis model, load the pre-trained weights, try to generate kaleidoscopic features, and further improve our methods with advanced mathematical tools. Specifically, we used the pre-trained VGG-19 model as the core, extracting features from selected layers and computing the Gram matrices. Using a white noise image as input, we also passed it through the CNN and computed the loss function on the corresponding layers. The weighted loss from each layer was used as the final loss function, and gradient descent was performed to eventually find a new image that matches the Gram matrices of the original texture. In addition, we also implemented the CNNMRF model, which is also based on VGG-19 but further leverages MRF and EM algorithms. We conducted extensive experiments and comparisons, and explored control methods and styles.

## Parameters

| Argument            | Default Value                                  | Type  | Description                  |
|---------------------|------------------------------------------------|-------|------------------------------|
| `--model`           | `vgg19`                                        | `str` | The pre-trained model to use for texture synthesis.                  |
| `--gt_path`         | `leaf.jpg`                                    | `str` | path to ground truth image   |
| `--pool`            | `avg`                                          | `str` | pooling method               |
| `--rescale`         | `True`                                         | `bool` | rescale weights or not       |
| `--optimizer`        | `Adam`                                         | `str` | optimize method              |
| `--epoch`           | `1000`                                         | `int` | epoch                        |
| `--lr`              | `0.05`                                         | `float` | learning rate               |
| `--device`          | `cuda:0`                                       | `str` | device                       |
| `--save_path`       | `result.jpg`                                   | `str` | save path                    |
## Usage

To run the texture synthesis tool, use the following command:

```bash
python Synthesis.py --image_path path/to/your/image.jpg --style_image path/to/your/style.jpg
