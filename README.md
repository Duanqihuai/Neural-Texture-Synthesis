# Neural-Texture-Synthesis
## Parameters

- `--model`: (str, default: `'vgg19'`)  
  The pre-trained model to use for texture synthesis.

- `--pooling`: (str, default: `'avg'`)  
  The pooling method to apply (e.g., `'avg'` for average pooling).

- `--rescale`: (bool, default: `True`)  
  Whether to rescale the weights or not.

- `--lr`: (float, default: `0.1`)  
  The learning rate for the optimization process.

- `--image_path`: (str, default: `'images/pebbles.jpg'`)  
  The path to the source image for texture synthesis.

- `--output_filename`: (str, default: `'output.jpg'`)  
  The filename to save the synthesized image.

- `--epochs`: (int, default: `1000`)  
  The number of epochs to run the synthesis process.

- `--layer_list`: (list of str, default: `['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']`)  
  A list of layers in the model to use for texture synthesis.

- `--method`: (str, default: `'gram'`)  
  The method used for texture synthesis (you can choose 'cnnmrf' or 'gram').

- `--h`: (float, default: `0.5`)  
  A hyperparameter (`h lambdas`) used in the synthesis process.

- `--patch_size`: (int, default: `7`)  
  The size of the patches used in the synthesis process.

- `--lambda_orientation`: (float, default: `0`)  
  A weight (`lambda`) for orientation-related loss.

- `--lambda_occurrence`: (float, default: `0.05`)  
  A weight (`lambda`) for occurrence-related loss.

- `--lambda_colorstyle`: (float, default: `0`)  
  A weight (`lambda`) for color style-related loss.

- `--style_image`: (str, default: `'images/pebbles.jpg'`)  
  The path to the style image used for texture synthesis.

- `--target_orientation_file`: (str, default: `''`)  
  The path to a file containing target orientation information (if applicable).

- `--output_folder`: (str, default: `'./outputs'`)  
  The folder where the output files will be saved.

## Usage

To run the texture synthesis tool, use the following command:

```bash
python Synthesis.py --image_path path/to/your/image.jpg --style_image path/to/your/style.jpg
