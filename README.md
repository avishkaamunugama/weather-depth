# WeatherDepth: Monocular Depth Estimation in Adverse Weather Conditions with Adversarial Training

Estimating the depth from a single RGB image is a challenging task that finds applications in various fields such as autonomous driving, robotics, 3D modelling, scene understanding, etc. Though there has been plenty of research in recent years on this field, of which many have produced outstanding results on daytime clear weather conditions, very few researches have been carried out that aim to solve monocular depth estimation in adverse weather conditions. The lack of research in adverse weather monocular depth estimation systems and the inability of the current state-of-the-art approaches that produce remarkable results in clear daytime conditions to produce accurate and consistent results when subjected to adverse weather conditions such as rain, snow or fog has been a significant setback in the autonomous driving industry and has been a key factor forcing most autonomous driving companies to still focus on expensive sensor-based approaches. This dissertation presents WeatherDepth, a novel robust monocular depth estimation approach that is capable of producing high-resolution accurate depth maps in adverse weather conditions such as rain, snow, and fog with the help of transfer learning and the CityscapeWeather dataset, which is an adverse weather depth estimation dataset based on the popular Cityscape dataset. The proposed approach, WeatherDepth, utilizes an adversarially trained autoencoder-based architecture. Experiments on the CityscapeWeather and vKITTI datasets demonstrate that our approach outperforms the state-of-the-art monocular depth estimation systems in generalization capabilities when subjected to adverse weather conditions.

**KEYWORDS:** Monocular Depth Estimation, Generative Adversarial Network, Autoencoder, U-Net, Transfer Learning, Adverse Weather Conditions

## Proposed Architecture
<p align="center">
  <img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/1d3649b6f4bef526069932838b6b0be225754ea6/Images/ScreenShot_5.png" height="300">
</p>

### Results

**CityScapesWeather Dataset** (Available at: https://drive.google.com/file/d/1eQM21gM6CdnOO-4OsnRljn0InUxusAg2/view?usp=sharing)

<img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/0385a6f58c746ed01da81f9f0a2813d959175e05/Images/ScreenShot_6.png" width="750">

**vKITTI Dataset** (Available at: https://drive.google.com/file/d/134ONfS-bMWMCDb7m62sDaD4fZwNaxejK/view?usp=sharing OR https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)

<img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/0385a6f58c746ed01da81f9f0a2813d959175e05/Images/ScreenShot_7.png" width="1000">

### MacOS Application UI
<p align="center">
  <img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/7f14aaec2404845559276315cdbf19368b679d56/Images/ScreenShot_1.png" height="205">  <img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/7f14aaec2404845559276315cdbf19368b679d56/Images/ScreenShot_2.png" height="205">
  <img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/7f14aaec2404845559276315cdbf19368b679d56/Images/ScreenShot_3.png" height="400">
  <img src="https://github.com/avishkaamunugama/WEATHERDEPTH/blob/7f14aaec2404845559276315cdbf19368b679d56/Images/ScreenShot_4.png" height="400">
</p>

## Complete project source code, documentation, test results, and project demonstration video : https://drive.google.com/file/d/1EWnobSWZ2t9l52saB35jRNYGnzxH8meR/view?usp=sharing

