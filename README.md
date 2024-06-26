# 泰拉知识问答系统  
***
项目介绍
---
本项目主要内容为使用自制数据集微调后的模型通过知识库检索进行游戏《明日方舟》内容的知识检索问答系统。    
【注：】使用自制数据集进行LLM微调的过程在上一篇文章：[【LLM微调】自制数据集：LLaMa Factory LLM微调教程](https://aistudio.baidu.com/projectdetail/8062297)
  
安装
---
```
// 克隆项目
git clone https://github.com/longkong39/longkong39-TaiLa_KnowledgeQuizSystem.git
```
```
// 手动修改项目文件夹名称为TaiLa_KnowledgeQuizSystem
```
```
// 安装环境
cd TaiLa_KnowledgeQuizSystem
pip install -r requirements.py
// 安装nltk所需内容
python
import nltk
nltk.download("punkt")
exit()
```
```
//下载模型
//embedding模型：https://aistudio.baidu.com/datasetdetail/255052
//微调后的LLM模型：https://aistudio.baidu.com/datasetdetail/278810
//下载后的文件分别放置到Embeddings和llm文件夹下
├── PRTS
  └── models
    ├── Embeddings
    └── llm
```
  
项目目录介绍
---  
![File_Tree_Description](https://github.com/longkong39/longkong39-TaiLa_KnowledgeQuizSystem/assets/109353411/f29a9ffd-5960-472e-b08a-cbf677730131)

执行
---
```
streamlit run webui.py
```

效果
---
![homepage](https://github.com/longkong39/longkong39-TaiLa_KnowledgeQuizSystem/assets/109353411/fc35057f-148f-43cc-a2e3-ef9549b76240)
![talk](https://github.com/longkong39/longkong39-TaiLa_KnowledgeQuizSystem/assets/109353411/1c2ce5b5-b956-413d-ad6b-f7dec9253c0f)
![knowledge_db](https://github.com/longkong39/longkong39-TaiLa_KnowledgeQuizSystem/assets/109353411/7a68c746-a465-4dc4-b73d-7d8b8dba1ae6)

图片版权与致谢
---
本项目中部分图片来源于网络。如果您发现任何图片侵犯了您的版权，请及时联系我们，我们会尽快核实并进行相应的处理，包括但不限于移除侵权内容。我们尊重所有原创者的劳动成果，并对您可能遇到的不便表示歉意。
