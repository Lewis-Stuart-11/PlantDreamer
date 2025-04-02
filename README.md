# PlantDreamer

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDcwZDFnMmZ4ZWZpandxbXE1enhmcTdrNDZ3ZTd1MW1mdDF4cm5qeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oeVGNAQiXR4KkGe0DZ/giphy.gif" width="40%">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXE3aW42Y21jaHY1MXZkdnoyOGk0ZGcxZHllMXhoejhsaXVsanNucCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kKwMwfFkEmZHXZr5aP/giphy.gif" width="40%" style="margin-left:2%">
</p>
<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3NrbmZybWpjZXRlNTdscXNmamVrcG5zZ2lsNzI1bHNwcGp2YXRqcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Tnz4f6MHjG6ylsdlKi/giphy.gif" width="45%">
</p>

Generation of 3D plant models is difficult, with many popular text-to-3D models failing to capture the intricate geometry and texture of real plant species. PlantDreamer is new framework that can generate 3D plants based either on a synthetic L-System mesh, or a real plant point cloud. We currently support generation of *bean*, *kale* and *mint* plants. 

Full explanation of this process can be found on our **[paper]()** .

Credit to the following repositories that were used as part of this codebase: [GaussianDreamer](https://github.com/hustvl/GaussianDreamer) and the GaussainDreamer [threestudio extension](https://github.com/cxh0519/threestudio-gaussiandreamer).

## Installation

### ThreeStudio Custom Extension
First, you will need to install Threestudio, an advanced framework for 3D object generation. Follow the tutorial on [ThreeStudio repository](https://github.com/threestudio-project/threestudio). This project has been tested on Threestudio version **0.2.3**.

Once installed successfully, navigate inside the Threestudio directory using the console, and enter the following:

```
cd custom
git clone https://github.com/Lewis-Stuart-11/PlantDreamer.git
cd PlantDreamer
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
git clone https://github.com/DSaurus/simple-knn.git
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
pip install open3d
pip install diffusers==0.30.3
pip install transformers==4.44.2
pip install bpy
pip install plyfile
pip install peft 
```

This installs PlantDreamer into ThreeStudio as a custom extension.

### Blender L-Systems 

If you want to run the L-Systems initialisation, you need to ensure that Blender is installed and added to you system path variable. We recommend using Blender **4.4**, but other modern versions should work fine. If ```Blender --version``` produces an error in your console, then you need to make sure Blender is added correctly to your system path variable.

In addition, the [Blender_L-Systems](https://github.com/krljg/lsystem) framework must be correctly installed as an add-on in your blender application. Please follow the steps on the native repository to correctly install this.

## How to Run

PlantDreamer offers two modes for training a 3DGS Plant. 

The first method involves generating a purely synthetic plant based on an L-System. If you have installed the 3D L-System add-on in Blender, then synthetic plants can be generated directly in the PlantDreamer pipeline. This can be trained using the following command:

```
python launch.py --config custom/PlantDreamer/configs/plantdreamer.yaml  --train --gpu 0 system.geometry.plant_type="bean" system.geometry.initialisation_type="l-system"
```

The other method generates a 3D plant based on an initial point cloud or mesh. This can be trained using the following command:

```
python launch.py --config custom/PlantDreamer/configs/plantdreamer.yaml  --train --gpu 0 system.geometry.plant_type="bean" system.geometry.initialisation_type="ply" system.geometry.geometry_convert_from="<path to point cloud/mesh>" 
```

The ```plant_type``` should be set to the type of species that you want to generate. We currently offer construction of three different species: 'bean', 'kale' and 'mint'.

If you are getting **out of memory errors**, then lower the number of initialised points via the *num_downsample_points* argument. 

## Dataset

Our dataset can be downloaded from our website [UoNPlantImages](https://plantimages.nottingham.ac.uk/datasets.html#plantdreamer)

We offer a complimentary dataset for testing and training of 3D plants. This dataset consists of point clouds created from real life captured plants, as well as synthetic counterparts generated using the 3D L-Systems. Furthermore, we provide the trained 3DGS plants for the synthetic L-System meshes, as well as the real life captured point clouds, for our standard experiments as well as the ablation studies (for testing how colour and point cloud quality impact the final output). Finally, we provide the ground truth images, masks and results for each of our experiments. 

This dataset has the following structure:

```
+-- Plants
|   +-- Plant #1
|   |   +-- Data  
|   |   |   +-- 3DGS # Point cloud reconstructed using 3DGS (high quality)
|   |   |   +-- MVS # Point cloud reconstructed using MVS (medium quality)
|   |   |   +-- SfM # Point cloud reconstructed using SfM (low quality)
|   |   |   +-- Synthetic Mesh (L-System mesh)   
|   |   +-- PlantDreamer 
|   |   |   +-- 3DGS (standard 3D plant from real point cloud)
|   |   |   +-- black (colour ablation study)
|   |   |   +-- white (colour ablation study)
|   |   |   +-- noise (colour ablation study)
|   |   |   +-- mvs (quality ablation study)
|   |   |   +-- sfm (quality ablation study)
|   |   +-- PlantDreamer-Synth 
|   |   |   +-- coloured (standard 3D plant from synthetic mesh)
|   |   |   +-- black (colour ablation study)
|   |   |   +-- white (colour ablation study)
|   |   |   +-- noise (colour ablation study)
|   |   +-- Gaussian-Splatting (reconstructed 3DGS plant)
|   |   +-- Images
|   |   |   +-- RGB 
|   |   |   +-- Masks
|   |   |   +-- Undistorted (ground truth images)
|   |   +-- Transforms
|   |   |   +-- Original (camera poses in JSON format)
|   |   |   +-- Undistorted (camera poses in COLMAP format)
  +-- Plant #2
  ...
  +-- Plant #21
+-- LoRA_imgs
|   +-- Bean
|   +-- Kale
|   +-- Mint
+-- experiment_results.xlsx (all results for the different plants)
+-- update_filepaths.py
```

To run each of the 3DGS models, the paths will need to be updated to where you have unzipped this dataset on your local file system. To do this, run the 'update_filepaths.py' script. It is important that the name of the root directory of the dataset (PlantDreamer) is not altered.

## Adding new Plant Species

To add a new plant species to PlantDreamer, a customised LoRA must be trained (to ensure effective texturing during training) and a new L-System grammar must be constructed.

### Plant Species Class

A new plant species class must be constructed that defines the properties of your new species. This includes: the prompt, lora name and l-system mesh generator. 

To add a new species, add a new 'PlantSpecies' subclass in 'plant_handler.py' and add this to the factory function at the bottom of the file. Make sure to set the string identifier as **your dedicated species name**.

### LoRA Training

To train a new LoRA, we recommend using [KoyaSS](https://github.com/bmaltais/kohya_ss) to perform this operation. This repo has tutorials that you can follow to training a LoRA for yourself. Make sure to train this for the *stable-diffusion-2-1-base* model (unless you decide to change the diffusion model). 

A dataset is required in order to train the LoRA. This dataset should consist of images (with filenames starting from 1 onwards) with matching text files which contain the description for the plant (with filenames that match the images, with a txt extension rather than the extension of the image). We recommend using around 20-30 images of each plant species from a range of different camera angles over a broad range of different plant morphologies. 

Once the LoRA is trained, place it into the LoRA directory along with the other trained LoRAs, with the **directory as the name of your dedicated species name**.

### L-System

To generate a new L-System grammar for your species, we recommend looking at the documentation on the [Blender_L-Systems](https://github.com/krljg/lsystem) repo. As a reference, each of the 'bean.py, 'kale.py' and 'mint.py' files in the 'l_systems' directory contain a 'generate_lsystem' function, which defines how the Blender L-System package should construct a new plant. We recommend getting a new plant L-System generated correctly in Blender first, before trying to get this integrated into PlantDreamer. 

In this Blender file, you will need to model the leaves and other foliage which will be used by the L-System grammar to generate a new plant structure, but will not be used as part of the final model. Make sure to put all of these objects inside of a **collection titled 'lsystem'**. Our script will remove all of these excess objects when converting the object to a '.ply' mesh. Objects such as the pot, soil and actual generated L-System plant should not be included in this collection, as these are part of the intended 3D model. For more information about this, we recommend inspecting the 'bean.blend, 'kale.blend' and 'mint.blend' files.

Once you have created a new plant L-System, add this Blender project file to the 'l_systems/blender_models' directory. Make sure to set the name of this Blender file as **your dedicated species name**. Next, create a new Python file that will be used for generating a new plant mesh in the 'l_systems' directory. Make sure to set the name of this python file as **your dedicated species name**. Create a new function that executes the l_systems logic and pass this as an argument to the 'generate_mesh' function in 'l_systems/scripts/generate_mesh.py'. For more examples of how this works, we recommend inspecting the 'bean.py, 'kale.py' and 'mint.py' files.

## Citation

```

```
