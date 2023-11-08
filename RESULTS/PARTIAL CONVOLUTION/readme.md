# Image Inpainting 

This comprehensive project combines various techniques for image inpainting . It includes Fast Marching Method (FMM) for image inpainting, and partial convolutions for image inpainting using Keras.

## Table of Contents

- [Image Inpainting using FMM](#image-inpainting-using-fmm)
  - [Description](#description-1)
  - [Usage](#usage-1)
  - [Dependencies](#dependencies-1)
- [Partial Convolutions for Image Inpainting](#partial-convolutions-for-image-inpainting)
  - [Description](#description-2)
  - [Usage](#usage-2)
  - [Dependencies](#dependencies-2)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Technologies Used

- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Google Colaboratory (Colab)**
- **TensorFlow**
- **PyTorch**
- **Kaggle**
- **Weights and Biases (Wandb)**

## File Structure:

📦IMAGE-INPAINTING
 ┣ 📂ASSETS
 ┃ ┣ 📂IMAGES
 ┃ ┃ ┣ 📜FMM_INPUT_IMAGE.jpg
 ┃ ┃ ┗ 📜PCONV_INPUT_IMAGE.jpg
 ┃ ┗ 📂MASKS
 ┃   ┣ 📜FMM_INPUT_MASK.png
 ┃   ┗ 📜PCONV_INPUT_MASK.png
 ┣ 📜PROJECT_REPORT.pdf
 ┣ 📂RESULTS
 ┃ ┣ 📂FMM
 ┃ ┃ ┣ 📜Screencast from 08-11-23 04...mp4
 ┃ ┃ ┗ 📜output.png
 ┃ ┗ 📂PARTIAL_CONVOLUTION
 ┃   ┣ 📜PCONVRESULT.png
 ┃   ┗ 📜Screencast from 08-11-23 04...mp4
 ┣ 📂SRC
 ┃ ┣ 📂MODEL
 ┃ ┃ ┣ 📜INPAINTING_MODEL.py
 ┃ ┃ ┣ 📜PCON2D.py
 ┃ ┃ ┣ 📜model_v2 (1).png
 ┃ ┃ ┗ 📜trainedmodel (1).h5
 ┃ ┗ 📂NOTEBOOKS
 ┃   ┣ 📜FMM_IMPLEMENTATION.ipynb
 ┃   ┣ 📜LICENSE_PLATE_DETECTION.ipynb
 ┃   ┣ 📜pconv-implementation (3).ipynb
 ┃   ┗ 📜MAIN.py

## Image Inpainting:

### Aim

The aim of this Image Inpainting project is to provide a robust solution for digital image restoration and manipulation. Our goal is to develop an automated system that can detect and remove unwanted objects or defects in images, and seamlessly fill in the missing regions with plausible background content. This technology can be highly beneficial in the following domains:

- **Content Editing**: Assisting graphic designers and content creators to easily remove unwanted elements from images, saving time and effort in manual editing.
- **Privacy Preservation**: Automatically detecting and removing sensitive information or objects from images to protect privacy before sharing them on public platforms.
- **Cultural Heritage**: Restoring old or damaged photographs and artworks, preserving historical and cultural heritage with minimal human intervention.
- **Retail and Real Estate**: Enhancing product images or real estate photos by removing distracting elements, providing cleaner and more attractive visuals for potential customers.

By leveraging advanced machine learning algorithms and deep learning techniques, our project aims not only to enhance the visual aesthetics of images but also to contribute to the broader field of computer vision and image processing by addressing challenges related to context-aware scene understanding and reconstruction.

![Example of the input image the mask and the corresponding output](IMAGE-INPAINTING/RESULTS/PARTIAL_CONVOLUTION/PCONVRESULT.png)

## Image Inpainting using FMM

### Description
This component of the project focuses on image inpainting using the Fast Marching Method (FMM). FMM is a powerful algorithm for filling in missing or damaged regions in images while preserving their structural and textural properties. The code for this implementation can be found in the 'fmm_inpainting' directory.

### Usage
To use the FMM inpainting code:
1. Navigate to the 'FMM_IMPLEMENTATION' file stored in SRC-NOTEBOOKS .
2. Copy the code give the image to be inpainted as the input and create a binary mask with damaged pixels as black and the rest as white and apply the algorithm 


### Dependencies
The FMM inpainting component relies on the following:
- Python
- Libraries specified in 'requirements.txt' within the 'fmm_inpainting' directory.

## Partial Convolutions for Image Inpainting

### Description
This component explores the use of partial convolutions for image inpainting using Keras. Partial convolutions are effective in preserving details during the inpainting process. The code for this implementation can be found in the 'partial_convolution' directory.

## Usage

To perform image inpainting using partial convolutions, follow these steps:

1. Create a Python file and load the provided `trained_model (1).h5` model stored in SRC-MODEL.

2. Add the definitions of `PConv2D` and `dice_coef` as shown below:

```python
   from tensorflow.keras.utils import get_custom_objects

   # Add batch dimension to input
   input_image = np.expand_dims(input_image, axis=0)
   input_mask = np.expand_dims(input_mask, axis=0)

   # Update custom objects
   get_custom_objects().update({'PConv2D': PConv2D, 'dice_coef': dice_coef})

   # Load your model
   model = load_model('/content/trainedmodel (1).h5')

   # Run the model to predict the inpainted image
   predicted_image = model.predict([input_image, input_mask])

   # Post-process the output
   output_image = predicted_image.squeeze()  # Remove batch dimension
   output_image = (output_image * 255).astype(np.uint8)
```
In this code snippet:

input_image and input_mask represent the image to be inpainted and its mask.
output_image contains the resulting inpainted image after processing.
Further Display the output image to see the results


### Dependencies

The partial convolutions for image inpainting component rely on the following:
- Python
- Libraries specified in 'requirements.txt' within the 'partial_convolution' directory.

## Future Work

Our future work will harness Generative Adversarial Networks (GANs) to elevate image inpainting, especially for images with extensive damage. GANs, with their innovative generative-discriminative interplay, hold the potential to craft detailed, context-aware fill-ins. We’ll focus on perfecting GANs for complex areas, enhancing texture blending and edge coherence where partial convolutions fall short.

We plan to tailor GANs for precise detail recovery in intricate or semantically significant regions. Our goal is a system that delivers seamless, undetectable inpainting across any image, which could revolutionize fields from art restoration to medical imaging. Integrating GANs represents a leap toward indistinguishable, realistic image reconstruction, propelling automated image restoration forward.

## Acknowledgments

We would like to express our gratitude for the tools and platforms that contributed to the success of this project:

- **Keras**: We leveraged the Keras deep learning framework for implementing partial convolutions in image inpainting.
- **Google Colab**: Google Colab provided a collaborative environment for the development of the FMM and object detection components. Its cloud-based Jupyter notebooks facilitated data processing and model training.
- **Kaggle**: Kaggle served as our platform of choice for collaborative work on the partial convolution part of the project. Its collaborative coding environment and version control features streamlined the development process.

## Contact

If you have any questions, need assistance, or wish to get in touch with us, please feel free to reach out to the authors:

- **Kindipsingh Mallhi**: [GitHub](https://github.com/kindipsingh)
- **Aditi Dhumal**: [GitHub](https://github.com/aditidhu)

