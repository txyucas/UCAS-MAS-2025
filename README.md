# UCAS-MAS-2025
## <p align="right">中国科学院大学2025年多智能体博弈小组作业</p>
## 仓库说明：
1. 本仓库用于存放中国科学院大学2025年多智能体博弈小组作业，包括代码、文档等，模型未上传到仓库中。
2. 本项目依赖于及第ai，参与[国科大多智能体系统课程作业2025擂台赛](http://www.jidiai.cn/compete_detail?compete=57)。
3. 本项目使用[AI-奥林匹克引擎](https://github.com/jidiai/olympics_engine)，使用前请前往前往该仓库配置环境

## 使用说明
1. 下载奥林匹克引擎
``` 
conda create -n MAS
conda activate MAS
git clone https://github.com/jidiai/olympics_engine.git
cd olympics_engine 
pip install .
```
2. 配置本项目的环境
```
cd UCAS-MAS-2025
pip install -r requirements.txt
```
3. 训练模型
```
ToDo
```
4. 测试模型
```
ToDo
```
5.可视化
```
python olympics_engine\olympics_engine\main.py #奥林匹克引擎自带的可视化工具
python visualize_no_note.py #在其基础上精简的可视化工具

```

## 项目结构
- agents: 包含智能体实现，目前有笔直行走的runing的文件
- algorithm: 包含算法实现，目前有都为空，预期加入HAC算法和PPO算法
- configs: 配置文件目录，包含 config.json(及第ai的envs自带的config) 和 config.py(模型训练时用到的config，包含模型参数等，目前为空)
- envs: 包含多个奥林匹克游戏环境实现，如足球、冰球等，从及第ai获取，项目中不会对其操作
- models: 神经网络模型目录，包含 CNN 和 FNN 实现，已经实现，后续更新config
- olympics_engine: 奥林匹克引擎，项目中基本不会对其进行操作 
- references: 参考资料和示例代码,仅供参考使用，最后会删掉，参考[easy-rl](https://github.com/datawhalechina/easy-rl)
- 根目录下的主要文件:
  - eval.py: 评估脚本，目前为空
  - train.py: 训练脚本 ，目前为空
  - visualize.py/`visualize_no_note.py`: 可视化工具
  - `submission.py`: 提交文件，目前为从及第ai获取的示例文件
  - requirements.txt: 依赖包列表，后续更新



## Cite
```
@misc{UCAS-MAS-2025,
  author = {tianxiangyu@txyucas,geqiyu@},
  title = {中国科学院大学2025年多智能体博弈小组作业},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```