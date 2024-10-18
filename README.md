# Diffusion Model for Digital Rock Images
Welcome to the official GitHub repository for the paper titled "**Enhancing Digital Rock Analysis through Generative Artificial Intelligence: Diffusion Models**." This repository hosts the generative artificial intelligence models that were developed and generated as part of the research. Additionally, you'll find the inference code for utilizing these trained models.

## Installing the Requirements and Running the Inference Script
**1. Setting Up Environment**

Before running the inference script, ensure you have the required dependencies installed. You can install them using the following command:

<div id="codeSnippet">
  <pre><code>pip install -r requirements.txt</code></pre>
  <button onclick="copyCode('pip install -r requirements.txt')"></button>
</div>

This command will install all the necessary Python packages listed in the requirements.txt file.

**2. Running the Inference Script**

Once you have installed the dependencies you can run the inference script using the following command:

<div id="codeSnippet">
  <pre><code>python inference.py --type &lt;MATERIAL_TYPE&gt; --num_generate_images &lt;NUM_IMAGES&gt; --num_loop &lt;NUM_LOOPS&gt;</code></pre>
  <button onclick="copyCode('python inference.py --type &lt;MATERIAL_TYPE&gt; --num_generate_images &lt;NUM_IMAGES&gt; --num_loop &lt;NUM_LOOPS&gt;')"></button>
</div>

Replace <MATERIAL_TYPE>, <NUM_IMAGES>, and <NUM_LOOPS> with your desired material type (sandstone or carbonate), the number of images to generate at each loop, and the number of loops, respectively.

For example:

<div id="codeSnippet">
  <pre><code>python inference.py --type sandstone --num_generate_images 8 --num_loop 1</code></pre>
  <button onclick="copyCode('python inference.py --type sandstone --num_generate_images 8 --num_loop 1')"></button>
</div>

This command will generate images using the specified model and parameters, saving them in the "./Generated_images" directory.

**3. Accessing Generated Images**

After running the inference script, you can find the generated images in the "./Generated_images" directory. Additionally, the script will compress the images into a zip file named Generated_images.zip for easier distribution and storage.

That's it! You have successfully run the inference script to generate images using the diffusion model. Feel free to explore and analyze the generated images for your digital rock analysis needs.

**Note:** if you encounter the following error:
<div id="codeSnippet">
  <pre><code>Access denied with the following error:
 	Cannot retrieve the public link of the file. You may need to change
	the permission to 'Anyone with the link', or have had many accesses. 
You may still be able to access the file from the browser:
	 https://drive.google.com/uc?id=1rifCP9gTgoBobhMugEuhtUcMfWYTpt62 </code></pre>
</div>

## Examples of Real and Generated Images

<table align="center">
  <tr>
    <td style="text-align: center;">
      <div>
        <img src="Images/8.png" alt="Sandstone Images">
        <figcaption>Sandstone Images</figcaption>
      </div>
    </td>
    <td style="text-align: center;">
      <div>
        <img src="Images/9.png" alt="Carbonate Images">
        <figcaption>Carbonate Images</figcaption>
      </div>
    </td>
  </tr>
</table>

## Citation

<div id="citation">
  <pre><code>@article{ESMAEILI2024127676,
  title = {Enhancing digital rock analysis through generative artificial intelligence: Diffusion models},
  journal = {Neurocomputing},
  volume = {587},
  pages = {127676},
  year = {2024},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2024.127676},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231224004478},
  author = {Mohammad Esmaeili},
}</code></pre>
  <button onclick="copyCitation()"></button>
</div>
