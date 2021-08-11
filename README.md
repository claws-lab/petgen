## PETGEN: Personalized Text Generation Attack on Deep Sequence Embedding-based Classification Models (ACM SIGKDD 2021)

#### Authors:  [Bing He](https://www.cc.gatech.edu/people/bing-he), [Mustaque Ahamad](https://www.cc.gatech.edu/~mustaq/), [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)

#### [Link to the paper](https://www.cc.gatech.edu/~srijan/pubs/petgen-he-kdd2021.pdf)

### Introduction
What should a malicious user write next to fool a detection model?
Identifying malicious users is critical to ensure the safety and integrity
of internet platforms. Several deep learning based detection
models have been created. However, malicious users can evade deep
detection models by manipulating their behavior, rendering these
models of little use. The vulnerability of such deep detection models
against adversarial attacks is unknown. Here we create a novel
adversarial attack model against deep user sequence embeddingbased
classification models, which use the sequence of user posts
to generate user embeddings and detect malicious users. In the
attack, the adversary generates a new post to fool the classifier.
We propose a novel end-to-end Personalized Text Generation Attack
model, called PETGEN, that simultaneously reduces the efficacy
of the detection model and generates posts that have several key
desirable properties.

![PETGEN](./visual/attack-setting.png)


If you make use of this code, the PETGEN algorithm, or the datasets in your work, please cite the following paper:
```
 @inproceedings{he2021petgen,
	title={PETGEN: Personalized Text Generation Attack on Deep Sequence Embedding-based Classification Models},
	author={He, Bing and Ahamad, Mustaque and Kumar, Srijan},
	booktitle={Proceedings of the 27th ACM SIGKDD international conference on Knowledge discovery and data mining},
	year={2021},
	organization={ACM}
 }
```
### Data
Data: the data is presented as follows: (Here, we take a sequence with 3 posts as an example)
- Sequence = (post1, post2, post3)
- Context = (context1, context2, context3)
- Label = 0 OR 1 (0: benign 1: malicious)

Then we save it in the dictionary by pickle file as follows:
- Seq2context: {(post1, post2, post3): (context1, context2, context3)}
- Seq2label: {(post1, post2, post3):label}
- Here is the [link](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fsh%2Fc7cazrvmgnq8q9s%2FAABNSroxV9CkPM88zUzhAan7a%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNGiDGaVZO4jBSB-We-5ySDief_mxw).

### Code

To run the code, go to "run" directory and use the following command line:
```angular2html
bash petgen.sh
```
For the package support, please run: 
```angular2html
pip install -r requirements.txt
```
- if you have any questions, please feel free to contact Bing He (bhe46@gatech.edu).
- if you have any suggestions to make the release better, please feel free to send a message.
- our code is based on [Text-GAN](https://github.com/williamSYSU/TextGAN-PyTorch) repository (Many thanks). If possible, please make sure Text-GAN can be executable at first.