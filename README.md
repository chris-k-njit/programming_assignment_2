# Module 07 - Assignment 01: Programming Assignment 2 - CS643852
## Christopher Keddell, Spring 2025

### Project Overview
This programming assignment tasks students like myself to build a machine learning (ML) model to predict wine quality by using Apache Spark's MLlib.

Included in this is:
- Parallel training over 4 AWS EC2 instances
- Parameter tuning and model validation
- An application used for these predictions, wrapped within a Docker container to have easier deployment

Python was chosen for this solution when using Apache Spark, and dealing with Docker container to ensure a seamless experience across different environments.

### Project Structure
- Training data in the TrainingDatasety.csv file
- Validation data within the ValidationDataset.csv file
- requirements.txt showing all python package dependencies
- Dockerfile showing instructions on how to containerize this app
- Training code within training_model.py when training an ML model via Spark

#### Best ways of Training the Model in Parallel on AWS
1. Create/launch 4 EC2 instances on AWS (ideally Ubuntu 22, or AMI)

2. Ensure Java, Spark and Python packages are fully installed

3. Be sure to upload project files to the master node

4. Next, start spark in standalone cluster mode
    - Use a shell command on master
    - Then, confirm the worker nodes have joined the cluster

5. Execute and run the application with this code:
```shell
spark-submit --master spark://<master-node-private-ip>:7077 training/training_model.py
```

6. Now that model is trained, it can be saved for usage in the future.


#### Build then run the Docker container 
1. Build the Docker Image:
```shell
docker build -t username/imagename:latest .
```

2. Now push the image to DockerHub:
```shell
docker push username/imagename:latest .
```

3. Finally, pull and run the Container (either on a local machine or via a single EC2 instance):
```shell
docker pull username/imagename:latest
docker run --rm username/imagename:latest
```

All is working as expected should the container execute the prediction model and print out a F1 score as the output.

##### Requirements for Project
- Python 3.8 or newer
- Apache Spark 3.4.0 or newer
- Java 8 or 11
- Docker

Python packages installed by
```python3
pip install -r requirements.txt
```

###### AWS Setup Instructions
1. Install Java
```shell
sudo apt update
sudo apt install openjdk-11-jdk
```

2. Install Apache Spark
```shell
wget https://downloads.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
tar -xvzf spark-3.4.0-bin-hadoop3.tgz
export SPARK_HOME=~/spark-3.4.0-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH
```

3. Install Python packages
```shell
pip install pandas numpy scikit-learn pyspark
```

4. Duplicate/copy to all EC2 instances as needed

5. Start the Spark Cluster
```shell
$SPARK_HOME/sbin/start-all.sh
```

##### F1 Score
- Model performance based on F1 score via validation dataset provided. 
- A final F1 score generated and printed during execution.