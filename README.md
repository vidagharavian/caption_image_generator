<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- ABOUT THE PROJECT -->
## Generating images from caption

In this project ,we have worked on multiple classification tasks inorder to  achieve image generation from given captions.
The system serves the following three purposes:
* Preparing a data set of captions using **bag-of-words** and classify them by **Random Forest** 
    * The given data set have been consisted of multiple classes inorder to build a model using random forest we have picked bag-of-words technique to prepare dataset before process of modeling
    * for each part of this steps the further explanation has been added in jupiter file in caption/modeling.
* build a ResNet model for image dataset
* at the end with given caption first we will find the related label scores base on the first model and then find 5 nearest image to that labels using genetic algorithm. 
      
### Built With

* [Python](https://www.python.org)
* [numpy](https://numpy.org)
* [tensorflow](https://www.tensorflow.org)
* [Keras](https://keras.io)
* [Pandas](https://pandas.pydata.org)



<!-- GETTING STARTED -->
## Getting Started
before starting this project there are multiple requirement that you should install each three part of this project have their own requirements .
### Prerequisites
all the requirements of this project are in requirements.txt
* pip
  ```sh
  pip install -r requirements.txt
  ```
