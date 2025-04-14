import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
from olympics_engine.generator import create_scenario
from olympics_engine.AI_olympics import AI_Olympics
from olympics_engine.scenario import *

import torch
import torch.nn as nn
import random

# 在文件开头添加设备检查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_batch(batch_size, section=None):
    """
    Generates a batch of games.
    """
    games = []
    for _ in range(batch_size):
        max_step = 400
        vis = 200
        vis_clear = 5
        
        if section == 1:
            billiard_gamemap = create_scenario("billiard-competition")
            for agent in billiard_gamemap['agents']:
                agent.visibility = vis
                agent.visibility_clear = vis_clear
            game = billiard_competition(billiard_gamemap)
        elif section == 2:
            tablehockey_gamemap = create_scenario("table-hockey")
            for agent in tablehockey_gamemap['agents']:
                agent.visibility = vis
                agent.visibility_clear = vis_clear
            game = table_hockey(tablehockey_gamemap)

        elif section == 3:
            football_gamemap = create_scenario('football')
            for agent in football_gamemap['agents']:
                agent.visibility = vis
                agent.visibility_clear = vis_clear
            game = football(football_gamemap)

        elif section == 4:
            wrestling_gamemap = create_scenario('wrestling')
            for agent in wrestling_gamemap['agents']:
                agent.visibility = vis
                agent.visibility_clear = vis_clear
            game = wrestling(wrestling_gamemap)

        elif section == 5:
            curling_gamemap = create_scenario('curling-IJACA-competition')
            for agent in curling_gamemap['agents']:
                agent.visibility = vis
                agent.visibility_clear = vis_clear
                curling_gamemap['env_cfg']['vis'] = vis
                curling_gamemap['env_cfg']['vis_clear'] = vis_clear
            game = curling_competition(curling_gamemap)

        elif section >= 6:
            running_Gamemap = create_scenario("running-competition")
            game = Running_competition(running_Gamemap,map_id=section-5, vis = vis, vis_clear=vis_clear, agent1_color = 'light red', agent2_color='blue')
        elif section==None:
            game = AI_Olympics(random_selection=False, minimap=False)
        games.append(game)
    return games

class Classify(nn.Module):
    def __init__(self, input_size:int, hidden_size:int=80, output_size:int=10):
        """
        Initializes the neural network.
        :param input_size: Number of input features，sequeuen_len*chennel*height*width
        :param hidden_size: Number of neurons in the hidden layer
        :param output_size: Number of output classes+1
        """
         # Initialize the parent class
         # Initialize the layers
        super(Classify, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        # 将模型移至GPU
        self.to(device)
        
        

    def forward(self, x):
        '''
        x: Input data (batch_size,sequeuen_len,chennel,height,width)
        :return: Output of the neural network
        '''
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def predict(self, x):
        """
        Predicts the class of the input data.
        :param x: Input data
        :return: (predicted class, probability)
        """
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device)
            output = self.forward(x)
            _, predicted = torch.max(output.data, 1)
            probability = output.data[predicted]
        return predicted.item(), probability.item()
    def train(self, data, epochs=100, learning_rate=0.001):
        """
        Trains the neural network.
        :param data: Training data and labels
        :param data: (features, labels)
        :param epochs: Number of epochs
        :param learning_rate: Learning rate
        """
        features, labels = data
        # 将数据移至GPU
        features = features.to(device)
        labels = labels.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(features)
            # 添加检查
            if torch.any(outputs.argmax(dim=1) >= 10):
                print("Warning: Invalid predictions found:", outputs.argmax(dim=1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Training complete. Final loss: {loss.item()}")
        
    def __call__(self, x):
        """
        Calls the forward method of the neural network.
        :param x: Input data
        :return: Output of the neural network
        """
        return self.forward(x)
ID_to_name = {
    0: "None",
    1: "running-competition-1",  # running map 1
    2: "running-competition-2",  # running map 2
    3: "running-competition-3",  # running map 3
    4: "running-competition-4",  # running map 4
    5: "table-hockey",
    6: "football",
    7: "wrestling",
    8: "curling-competition",
    9: "billiard"
}

Name_to_ID = {
    "None": 0,
    "running-competition-1": 1,
    "running-competition-2": 2,
    "running-competition-3": 3,
    "running-competition-4": 4,
    "table-hockey": 5,
    "football": 6,
    "wrestling": 7,
    "curling-competition": 8,
    "billiard": 9
}
sequence_line = 1
classify = Classify(input_size=40*40*sequence_line)
# Load the model
model_path = "classify-ckpt/model_final.pth"
# 加载模型时也需要指定设备
if Path(model_path).exists():
    classify.load_state_dict(torch.load(model_path, map_location=device))
       
def get_classify(obs):
    """
    Classifies the given observation sequences.
    :param obs:  observation sequences of one agent
    :return: (predicted class:str, done:bool)
    """
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = obs.view(obs.size(0), -1)
    predicted_class, probability = classify.predict(obs)
    if predicted_class == 0 or probability < 0.5:
        done = False
    else:
        done = True
    return ID_to_name[predicted_class], done

def train(epochs:int=1000, batch_size:int=1024, dir='classify-ckpt',lr=0.001):
    """
    Trains the neural network.
    :param epochs: Number of epochs
    :param batch_size: Size of each batch
    """
    for epoch in range(epochs):
        data = get_batch(batch_size)
        origin_feature = [d.reset() for d in data]    
        
        # 修改标签生成逻辑
        labels = []
        for d in data:
            if isinstance(d.current_game, Running_competition):
                # running游戏根据map_index确定标签
                label = d.current_game.map_index
            else:
                # 其他游戏类型按原来的映射+4（因为前4个被running占用）
                game_name = d.current_game.game_name
                label = Name_to_ID[game_name]
            labels.append(label)
            
        labels = torch.tensor(labels).to(device)
        features_1 = torch.tensor([d[0]['agent_obs'] for d in origin_feature]).to(device)
        features_2 = torch.tensor([d[1]['agent_obs'] for d in origin_feature]).to(device)
        features = torch.cat((features_1, features_2), dim=0)
        labels = torch.cat((labels, labels), dim=0).to(device)
        classify.train((features, labels), epochs=1, learning_rate=lr)
        lr=lr*0.95
        print(f"Epoch {epoch+1}/{epochs} complete.")
        if (epoch+1) %1==0:
            #eval
            data = get_batch(batch_size)
            origin_feature=[d.reset() for d in data]    
            labels = []
            for d in data:
                if isinstance(d.current_game, Running_competition):
                    # running游戏根据map_index确定标签
                    label = d.current_game.map_index
                else:
                    # 其他游戏类型按原来的映射+4（因为前4个被running占用）
                    game_name = d.current_game.game_name
                    label = Name_to_ID[game_name]
                labels.append(label)

            labels = torch.tensor(labels).to(device)
            features_1 = torch.tensor([d[0]['agent_obs'] for d in origin_feature]).to(device)
            features_2 = torch.tensor([d[1]['agent_obs'] for d in origin_feature]).to(device)
            features = torch.cat((features_1, features_2), dim=0)
            labels = torch.cat((labels, labels), dim=0).to(device) 
            correct=0
            total=0
            with torch.no_grad():
                outputs = classify(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for (label, pred) in zip(labels, predicted):
                    if label != pred:
                        print(f"Correct: {ID_to_name[label.item()]}, Predicted: {ID_to_name[pred.item()]}")
                        print(f"Wrong: {ID_to_name[label.item()]}, Predicted: {ID_to_name[pred.item()]}")
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.2f}%")
            dir=dir
            if not Path(dir).exists():
                Path(dir).mkdir(parents=True, exist_ok=True)
            torch.save(classify.state_dict(), f"{dir}/model_epoch_{epoch}.pth")
            print(f"Model saved at epoch {epoch}.")
    print("Training complete.")
    torch.save(classify.state_dict(), f"{dir}/model_final.pth")
    print("Final model saved.")
if __name__ == "__main__":
    # Example usage
    # train(epochs=1000, batch_size=64)
    classify= Classify(input_size=40*40*1)
    
    train(epochs=100, batch_size=64)
