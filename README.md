Car Damage Detection with Image Recognition

Car Damage Detection :
TEAM MEMBERS: -

VIVEK CHINTA
2.PRARTHANA BAHURIYA

3.SUKUMAR CHIGURUPATI

INTRODUCTION
To detect/recognize dents, scratches, crushed, broken, and no - damage from images of cars. The dataset contains car images with one or more damaged parts. The folder has all 1533 images in the dataset. There are three more folders train/, val/ and test/ for training, validation and testing purposes respectively. Using this data for training a object detection model to identify and classify all the classes using the YOLOv7 model which is extensively used in autonomous vehicles and vehicle self-damage detection.

In the rage of emerging automotive technologies, many real-world infotainment systems have been developed for automotive with the help of machine learning techniques. One such invention is autonomous cars, autonomous cars in the sense an automatic car which makes its own decision based on experience. As a diagnosis system in autonomous cars a self-damage identifying system is essential during any external disturbance. Our proposed system identiﬁes and classiﬁes the Dent and Scratch in the car using Object Detection Techniques. Dents and Scratches can be identiﬁed at any part of the car using the ML models which will be further used as a test case in Car Damage Detection system and can be employed in the car dashboard as an notiﬁcation during external disturbance.

RESEARCH QUESTIONS
How does the performance of YOLO v7 compare to earlier versions in detecting car damage?

What impact does the size of the training dataset have on the accuracy of car damage detection using YOLO v7?

Can transfer learning effectively enhance the training efficiency and accuracy of YOLO v7 for car damage detection?

How does the model handle different types and severities of car damage, and what are the key challenges in classification?

Explanation of Real-World Implementation:
For starters, the system could revolutionize the way insurance companies assess car damage. Typically, this process is manual and can be time-consuming and subjective. With YOLOv7-based system, insurers could automate and standardize damage assessment. We upload pictures of the damaged car, and boom – the system provides a quick, detailed analysis of the damage. This would speed up claim processing times, enhance accuracy, and potentially even reduce fraud.

Another cool application could be in the automotive resale market. The system could be used to assess used cars, providing potential buyers with an unbiased, comprehensive report of any existing damage. This transparency could build trust in the market and might even help in accurately pricing the cars.

DATA TO BE USED
First Dataset

https://www.istockphoto.com/search/2/image-film?family=creative&phrase=car%20damage this dataset likely contains a collection of images related to car damage. Such datasets are usually used for visual recognition tasks in machine learning and artificial intelligence, especially in projects like the one involving YOLOv7 for damage assessment. These images would typically showcase various types and degrees of damage to vehicles, which could be invaluable for training a model to accurately identify and classify car damage in real-world scenarios.

Second Dataset:-

https://drive.google.com/drive/folders/1gv0nRUWok3ajOKIhAy87RFPndczcxftI?usp=drive_link

![0aa33c2d-125422955](https://github.com/vivekuk/Car_Damage_Detection/assets/60242797/a35aa295-661c-4fa8-8f3b-37f2e5e9a196)

 
Each entry consists of an image file name followed by a description of the damage, such as "A Car with a Scratch," "A Car with No Damage," "A Car with a Dent," "A Broken Car," and "A Crushed Car." This dataset is ideal for training a machine learning model like YOLOv7 for recognizing and classifying different types of car damage

APPROACH
•	Define Objectives and Scope:
Clearly define the objectives of your project. What specific types of car damage are you aiming to detect? Are you focusing on a specific dataset or real-world scenario?

•	Data Collection and Preprocessing:
Gather a diverse dataset containing images of cars with different types and severities of damage. Ensure the dataset is labeled with bounding boxes around the damaged areas. Preprocess the dataset by resizing images, normalizing pixel values, and augmenting data to increase variability.

•	Select or Prepare YOLO v7 Model:
Choose or train a YOLO v7 model suitable for object detection. You can start with a pre-trained YOLO v7 model on a general object detection dataset and fine-tune it on your car damage dataset.

•	Data Splitting:
Split your dataset into training, validation, and testing sets. This helps assess the model's performance on unseen data and prevents overfitting.

Training:
Train the YOLO v7 model on the training set. Experiment with hyperparameters, learning rates, and optimization techniques to achieve the best performance.

•	Validation and Hyperparameter Tuning:
Validate the model on the validation set to fine-tune hyperparameters and detect potential issues like overfitting. Adjust parameters such as learning rate, batch size, and model architecture based on validation performance.

•	Evaluation:
Evaluate the trained model on the testing set to measure its performance in terms of precision, recall, and F1 score. Analyze the model's ability to correctly detect and classify different types of car damage.

Project Responsibilities
Data Labeling:- Sukumar, Prarthana

Model building:- Vivek,Prarthana

data analysis:- Vivek,Sukumar



Project Overview:
The project aims to streamline the process of assessing car damage using image recognition. Traditional methods of manually inspecting cars for damage are time-consuming and resource-intensive. The proposed solution leverages image recognition to automate the assessment of car damage, providing a faster and more efficient alternative. This technology can be applied in real-world scenarios, with potential use cases in the insurance industry.

Features:
Image Recognition:

Utilizes advanced image recognition algorithms to identify and assess damage on cars.
Reduces manual effort and time required for damage assessment.
Accessibility:

Provides an easy-to-use interface for users to upload car images and receive instant damage reports.
Enhances accessibility and reduces the dependency on manual inspections.
Real-Time Use Case:

Demonstrates the practical application of the technology in a real-time use case.
Mimics scenarios where an insurance company, such as Molin's Insurance, can benefit from automated damage assessment.
Use Case Example:
Molin's Insurance:
Molin's Insurance, a hypothetical insurance company, can leverage the car damage detection system for various benefits:

Efficient Claims Processing:

Accelerates the claims processing workflow by quickly assessing the extent of car damage through automated image recognition.
Reduces the time customers need to wait for claim approvals.
Resource Optimization:

Minimizes the need for manual inspections and on-site visits, optimizing the allocation of human resources.
Improved Accuracy:

Enhances the accuracy of damage assessment by leveraging advanced image recognition algorithms, reducing the chances of human error.
Enhanced Customer Experience:

Provides customers with a seamless and efficient process for submitting and processing insurance claims.
Improves customer satisfaction through faster response times.
Technical Details:
Image Recognition Model:

Utilizes state-of-the-art convolutional neural networks (CNNs) for image recognition.
Trained on a diverse dataset of car images with and without damage.
Python Technologies:

Developed using Python programming language.
Utilizes popular deep learning libraries such as TensorFlow and Keras.
Web Interface:

Implements a user-friendly web interface for users to upload car images and receive damage reports.
Ensures ease of use for both insurance professionals and customers.

<img width="1036" alt="Screenshot 2023-12-11 at 8 41 49 PM" src="https://github.com/vivekuk/Car_Damage_Detection/assets/60242797/b8d544af-9028-436a-9bf5-18806bacca07">

Conclusion:

<img width="1042" alt="Screenshot 2023-12-11 at 8 42 04 PM" src="https://github.com/vivekuk/Car_Damage_Detection/assets/60242797/7b961e7d-67ee-4f71-9a62-a12028b3aa48">

The car damage detection system with image recognition presents a viable solution for automating the assessment of car damage. Its potential applications in the insurance industry, as demonstrated through the use case of Molin's Insurance, showcase the tangible benefits of adopting this technology. The project not only streamlines processes but also contributes to resource optimization and improved customer experiences.

By submitting this project, we aim to contribute to advancements in automated image recognition and its practical applications in real-world scenarios. The technology presented here aligns with the evolving landscape of artificial intelligence and its potential to transform traditional industries.
