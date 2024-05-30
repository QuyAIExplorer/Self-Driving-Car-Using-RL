### Autonomous Vehicle Control System using Deep Q-Network (DQN)
This project develops an autonomous vehicle control system using the Deep Q-Network (DQN) reinforcement learning algorithm. The goal is to train an agent to navigate a simulated racing environment and learn optimal driving policies. The project includes designing the racing environment, implementing the DQN algorithm in Python with TensorFlow, and fine-tuning hyperparameters for optimal performance.

### Objective:
The main objective of this project is to deepen my understanding of reinforcement learning and related algorithms. Additionally, I aim to enhance my Python coding skills, particularly in utilizing AI frameworks like TensorFlow.

### Files in the Project:
- GameEnv.py: Sets up the environment for training the car agent.
- Goals.py: Sets up the target award and punishment for the agent.
- Walls.py: Sets up walls so that if the car agent hits them, it will be punished.
- car.png: Image representing the agent on the GUI.
- ddqn_keras.py: Contains the Deep Q-Network algorithm.
- ddqn_model.h5: The file that will be trained and continuously updated.
- main.py: The main file for the agent to run in the environment and continuously update the agent's policy.
- main_test_model.py: After training, use this file to run the agent to the destination.
- track.png: Map showing the environment on the GUI.
- Report_Final_Project.docx: A report containing detailed information about the project and the reinforcement learning techniques used.
- Result_video.mp4: A video showing the car start running from start point to the goal
- Presentation_video: To understand more about each python file and the algorithm, you can refer to [this link](https://drive.google.com/file/d/1NnMlli14RS9sLOB_NRCIjbJofUeYJ8Wj/view?usp=sharing).

### Running the Project:
1. Clone the repository to your local machine.
2. Run main.py to start training the agent in the environment.
3. Once training is complete, use main_test_model.py to run the agent to its destination.
Feel free to explore and modify the code to suit your needs. If you encounter any issues or have any questions, please don't hesitate to reach out.

### Disclaimer: 
This project is for educational purposes only. The simulated environment does not accurately reflect real-world driving conditions, and the trained agent should not be used in actual autonomous vehicles.
