# Image Prediction Application

This application is a web service that receives an image, predicts its content using a machine learning model, and retrieves related information from Firebase.

**How it Works:**

1. An image is uploaded by the user via the `/predict` endpoint.
2. The application loads the machine learning model from Google Cloud Storage.
3. The image is preprocessed and normalized before being predicted by the model.
4. The prediction result is used to retrieve information from Firebase.
5. The information from Firebase is returned to the user as a JSON response.

**Technologies:**

* Python
* Flask
* TensorFlow
* Google Cloud Storage
* Firebase

**Deployment:**

This application is designed to be deployed in a cloud environment such as Google App Engine or similar services.

**Notes:**

* Ensure that you have configured your Google Cloud and Firebase credentials correctly.
* Adjust the file paths for `gcs_key.json` and `firebase_key.json` according to your configuration.
