# ChatYuan-lareg-v2: 元语功能型对话大模型v2

发布于：2023年3月23日

## 相关介绍

ChatYuan-large-v2是一个支持中英双语的功能型对话语言大模型。ChatYuan-large-v2使用了和 v1版本相同的技术方案，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

ChatYuan large v2 is an open-source large language model for dialogue, supports both Chinese and English languages, and in ChatGPT style.

ChatYuan-large-v2是ChatYuan系列中以轻量化实现高质量效果的模型之一，用户可以在消费级显卡、 PC甚至手机上进行推理（INT4 最低只需 400M ）。

<a href='https://huggingface.co/spaces/ClueAI/ChatYuan-large-v2' target="__blank">在线Demo(Huggingface)</a> &nbsp; | 
<a href='https://modelscope.cn/studios/ClueAI/ChatYuan-large-v2' target="__blank">在线Demo(ModelScope)</a> &nbsp; | 
  <a href='https://www.clueai.cn' target="__blank">使用API(large版)</a> &nbsp; | 
  &nbsp;<a href='https://colab.research.google.com/drive/1ZcLIJuemiojigrfjbsDMBWrX7JqXZX6I?usp=sharing' target="__blank">Colab在线试用</a> &nbsp; |
  &nbsp;<a href='https://mp.weixin.qq.com/s/FtXAnrhavA5u7hRyfm8j6Q' target="__blank">文章介绍</a> 
  
在chatyuan-large-v1的原有功能的基础上，我们给模型进行了如下优化：
- 增强了基础能力。原有上下文问答、创意性写作能力明显提升。
- 新增了拒答能力。对于一些危险、有害的问题，学会了拒答处理。
- 新增了中英双语对话能力。
- 新增了代码生成功能。对于基础代码生成进行了一定程度优化。
- 新增了表格生成功能。使生成的表格内容和格式更适配。
- 增强了基础数学运算能力。
- 最大长度从1024token数扩展到4096。
- 增强了模拟情景能力。

Based on the original functions of Chatyuan-large-v1, we optimized the model as follows:

-Added the ability to speak in both Chinese and English.

-Added the ability to refuse to answer. Learn to refuse to answer some dangerous and harmful questions.

-Added code generation functionality. Basic code generation has been optimized to a certain extent.

-Enhanced basic capabilities. The original contextual Q&A and creative writing skills have significantly improved.

-Added a table generation function. Make the generated table content and format more appropriate.

-Enhanced basic mathematical computing capabilities.

-The maximum number of length tokens has been expanded to 4096.

-Enhanced ability to simulate scenarios< br>


## 模型下载与体验地址

https://huggingface.co/ClueAI/ChatYuan-large-v2/

https://modelscope.cn/studios/ClueAI/ChatYuan-large-v2

## 声明
文本由模型生成的结果, 请谨慎辨别和参考, 不代表任何人观点


请在法律允许的范围内使用，详见[LICENSE](./LICENSE)

<a href='https://www.cluebenchmarks.com/clueai.html'>PromptCLUE-large</a>在1000亿token中文语料上预训练，累计学习1.5万亿中文token，并且在数百种任务上进行Prompt任务式训练。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对多种生成任务，可以进行采样自由生成。 

<a href='https://huggingface.co/spaces/ClueAI/ChatYuan-large-v2' target="__blank">在线Demo huggingface space </a> &nbsp; | 
  <a href='https://www.clueai.cn' target="__blank">使用API(large版)</a> &nbsp; | 
 &nbsp; <a href='https://github.com/clue-ai/ChatYuan' target="__blank">Github项目地址</a>&nbsp; |
  &nbsp;<a href='https://colab.research.google.com/drive/1ZcLIJuemiojigrfjbsDMBWrX7JqXZX6I?usp=sharing' target="__blank">Colab在线试用</a> &nbsp; |
  &nbsp;<a href='https://mp.weixin.qq.com/s/-axa6XcjGl_Koeq_OrDq8w' target="__blank">文章介绍</a> 






#### 代码范例

加载模型：
 
 ```python
# 加载模型
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2")
# 该加载方式，在最大长度为512时 大约需要6G多显存
# 如显存不够，可采用以下方式加载，进一步减少显存需求，约为3G
# model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2").half()



 ```

使用模型进行预测推理方法：
```python
# 使用
import torch
from transformers import AutoTokenizer
# 修改colab笔记本设置为gpu，推理更快
device = torch.device('cuda')
model.to(device)
def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t").replace('%20','  ')

def answer(text, sample=True, top_p=1, temperature=0.7):
  '''sample：是否抽样。生成任务，可以设置为True;
  top_p：0-1之间，生成的内容越多样'''
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 
  if not sample:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
  else:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0])
print("end...")
```

```python
input_text0 = "翻译这句话成英文：屈臣氏里的化妆品到底怎么样？"
input_text1 = "帮我写一个英文营销方案，针对iphone"
input_text2 = "写一个冒泡排序"
# input_text1 = "你能干什么"
# input_text2 = "用英文写一封道歉的邮件，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
input_text3 = "写一个文章，题目是未来城市"
input_text4 = "写一个诗歌，关于冬天"
input_text5 = "从南京到上海的路线"
input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
input_list = [input_text0, input_text1, input_text2, input_text3, input_text4, input_text5, input_text6, input_text7, input_text8]
for i, input_text in enumerate(input_list):
  input_text = "用户：" + input_text + "\n小元："
  print(f"示例{i}".center(50, "="))
  output_text = answer(input_text)
  print(f"{input_text}{output_text}")
```

## 预训练代码   
[text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer). 
## 微调代码  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lEyFhEfoc-5Z5xqpEKkZt_iMaojH1MP_?usp=sharing)
## 分布式训练
[分布式](https://github.com/clue-ai/ChatYuan/tree/main/distributed-training)

paddle相关使用和体验  
1. https://aistudio.baidu.com/aistudio/projectdetail/5452095  
2. https://aistudio.baidu.com/aistudio/projectdetail/5427279  

## 主要更新


### 增强了原有问答、生成与创意性写作功能
> 通过以下样例的对比，您可以感受到chatyuan-large-v2在上个版本上的升级和迭代
> 包括但不限于：
1. 生成内容更加丰富
2. 对用户的意图把握更加准确
3. 生成的内容也更为可靠，有条理性


- 写一封英文商务邮件给英国客户，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失


```
chatyuan-large-v1

Dear [Name],

I'm sorry to inform you that we have been doing our business to you for your delay. We understand that we can compensate you for your losses due to the delay in our logistics.

We understand that we can make sure we can compensate you for your losses due to the delay.

Thank you for your understanding and we are very grateful for your efforts.

Sincerely,

[Your Name]
```


```
chatyuan-large-v2

Subject: Due to delay in logistics, our company does not order the item as planned on time.

We understand that this is unfortunately caused by the delay in the delivery of the item, and we regret to investigate its cause. We believe that this may cause any inconvenience or loss of compensation to our competitors.

Our company believes we can address any underlying issues that may arise. We are confident that our services will be acceptable if any inconformity occurs.

Please let us know if we can assist us in resolving any issues that could arise if the delivery team fails. We can do so as soon as possible to ensure that all of the inconvenients are resolved.

Thank you for your understanding and support. We look forward to hearing from you soon.

Best regards,
```

- 写一个文章，题目是未来城市


```
chatyuan-large-v1


未来的城市是怎样的呢？  
  
未来城市拥有很多不同的功能，如智慧城市、智能交通系统、远程医疗系统、智能家居系统、智慧医疗系统等。未来城市还使用物联网技术，可以远程控制，使用远程操控和传感器来监控城市的发展，帮助他们解决各种城市问题。  
  
未来的城市还采用互联网技术，可以让城市变得更智能，让城市变得更智慧，让每一个人都能更轻松地生活。未来城市的智能设备也可以让城市的使用者更方便地掌握他们的生活，比如智能的智能交通系统和远程医疗系统和智能的安防系统等，可以提供他们更多的便利。  
总之，未来城市的愿景是，让所有城市的使用者都能拥有一个美好的未来，让他们可以在自己的城市享受美好的生活。  
```

```
chatyuan-large-v2


未来城市，是一场对未来的探索和探索，它将是一个充满机遇和挑战的时代。在未来城市中，人们将享受到更多便利和智慧，同时也将面临更多的挑战和机遇。

首先，未来城市的交通将更加智能化和高效化。因为城市中将不断发展和更新交通系统，并且交通信号将更加智能化和自动化，从而提高城市交通效率和安全性。同时，城市中的公共交通网络也将更加完善，人们可以更加轻松和便捷地到达城市各个角落，同时降低出行成本。

其次，未来城市的能源将更加清洁和可持续。城市中的能源消费将逐渐从化石燃料为主向可再生能源为主转变。未来城市的能源结构将更加多元化，将从传统的化石燃料为主向能源与能源的完美结合转变。同时，城市中也将采用更加环保的能源，并使用太阳能、风能等清洁能源，以降低对环境的污染和破坏。

此外，未来城市的医疗、教育等公共服务也将更加完善和智能化。城市中的医疗和教育设施将更加现代化和智能化，人们将能够在城市中心的医院中接受高品质的医疗服务。同时，城市将通过物联网技术，实现城市中的医疗和教育资源的共享，让每个孩子都有机会接受良好的教育。

最后，未来城市的生态环境将更加绿色和生态化。城市中的建筑和设施将更加注重环境保护和可持续发展。同时，城市也将采用更加环保和可持续的能源和交通方式，以减少对环境的影响。

总之，未来城市是一个充满机遇和希望的时代，它将为人们带来更多的便利和智慧，同时也将面临更多的挑战和机会。未来城市需要我们不断探索和创新，才能让这个美好的时代更加美好。
```

- 学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字

```
chatyuan-large-v1

在学前教育专业岗位实习中，在学生方面会存在一些问题，如学生缺少专业知识、缺乏合作意识、缺乏基本的责任心等等，需要我们采取措施，来解决这些问题。  
      
首先，在实习过程中，我们需要培养学生的团队意识，加强学生与老师之间的合作，加强与家长之间的沟通，提高他们的沟通能力。另外，我们也需要加强学生的独立性，让学生学会独立思考，学会和老师、家长进行沟通，培养其独立性。  
此外，在日常的工作中，我们也要为学生提供适当的机会和机会，让学生多接触社会、多参加社会活动，加深他们对社会、学校的认识，培养他们主动、积极和创新的精神。  
最后，我们还需要重视学生在实习中出现的问题，尤其是学生缺乏合作，有依赖心理，不懂得分享，有拖延行为等，这些学生应该引起我们的重视，制定合理的改进措施，改善这些问题。   
```

```
chatyuan-large-v2

一、改进措施

1. 建立科学严谨的教学管理制度。在学前教育专业岗位实习中，应该建立一套严谨的教学管理制度，注重培养学生实践能力和综合素质，提高学生的参与度和主动性。

2. 加强教师和学生的沟通。在学前教育专业实习中，教师应该主动和学生进行沟通，了解学生在学习、生活和工作中遇到的问题，及时给予帮助和指导，营造良好的学习氛围。

3. 提高学生的参与度。建议根据不同学生的特点，采取不同的教学方式和手段，充分调动学生的学习积极性和主动性，让他们在实践中学习，积累更多实践经验。

4. 加强学生自我管理能力。学前教育专业岗位实习过程中，学生需要学会自我管理，定期进行自我评估，及时发现和纠正学生的错误，保证实习效果。

5. 建立良好的考评机制。应该建立良好的考评机制，通过合理的考核和奖励机制，激励学生积极参与实习，不断提升实习质量。

二、总结

通过以上的改进措施，可以有效提高学前教育专业岗位实习的质量，让学生更好地理解和掌握所学知识，为学生未来的职业发展打下坚实的基础。
```

- 写一个诗歌，关于冬天

```
chatyuan-large-v1

冬夜寂静冷， 
云在天边飘，  
冰封白雪上，  
寒冷像一场雪。 
  
雪花融化成冰，  
像那雪花飘洒，  
在寒冷的冬天，  
感受春天的喜悦。  
  冬日里， 
风雪渐消，  
一片寂静，  
把快乐和温暖带回家。 
```

```
chatyuan-large-v2

白雪漫天飞舞， 
寒风刺骨寒冷。 
大地一片寂静， 
生命在寒冷中沉睡。 

枝头挂满雪花， 
寒气逼人让人心寒。 
然而， 
冬天的美景却让人难忘。 
因为有了它， 
冬天才有了生命的气息。
```


- 从南京到上海的路线

```
chatyuan-large-v1

你好，南京到上海的路线如下：  
      
1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。  
  
2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。  
  
3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站。    
```

```
chatyuan-large-v2

从南京到上海的路线：

1. 乘坐飞机：南京到上海的航班主要有南京禄口机场和上海浦东机场。南京禄口机场有航班到上海虹桥机场，浦东机场有航班到南京禄口机场。

2. 乘坐高铁：上海到南京的高铁线路有：上海虹桥-南京(沪宁高速-南京南站)-南京南站-上海虹桥(沪宁高速)-南京南站(沪宁高速——沪宁高速-沪宁高速-宁杭高速-宁杭高速公路-宁杭高速公路)-上海南站(沪宁高速公路)-上海虹桥(上海市区-上海南站)-上海虹桥站(上海市区-上海市区-浦东国际机场)。

3. 乘坐大巴：南京到上海，有多种交通工具可以选择。以下是其中几种：

1. 南京-上海的长途汽车和私家车，如K407、K451等。

2. 上海-南京的长途汽车和大巴，如K196等。

3. 上海-南京的公交车和私家车，如M29、M35等。

4. 乘坐出租车：上海到南京，有多种出租车可供选择。
```

- 根据标题生成文章：屈臣氏里的化妆品到底怎么样？

```
chatyuan-large-v1

近日，一条关于屈臣氏化妆品的微博引发网友热议。“在屈臣士商场里，有一些护肤品售价动辄上千元，部分化妆品甚至还卖到几折。屈臣仕化妆品专柜工作人员表示：“屈臣购商城的化妆品都是正规厂家生产，都是通过国家药品监督管理局的正规渠道进入市场的，并且都是正品。”“该微博发出后，引起网友们的热议，甚至不少网友开始搜索屈臣氏旗舰店来买产品。“屈氏化妆品真的值得买吗？“记者在屈士仕商场内看到，屈臣专卖店的货架上摆满了屈臣旗下的各种化妆品，有精华、面霜、眼霜等。““屈士士，这一款我买了，这款是主打美白的。”市民王女士说，她去年买了三瓶屈臣芮，其中一款是保湿面膜，还有一款是护手霜。“记者通过电话联系了屈臣臣芮的销售小姐，她说屈臣城里没有卖屈臣诗的化妆品，只有屈臣肌妍的化妆品。“
```

```
chatyuan-large-v2

作为一家以“健康、安全、有效”为理念的购物场所，屈臣氏里的化妆品备受消费者的喜爱。那么，屈臣氏里化妆品到底怎么样？

首先，屈臣氏的化妆品品种繁多，涵盖了各种类型和功效。例如，洁面乳、爽肤水、精华液、面霜、眼霜、唇膏等等。在选择时，消费者可以根据自己的肤质、需求和预算来选择适合自己的产品。

其次，屈臣氏内的化妆品价格通常比商场、超市等其他渠道优惠很多。所以，消费者可以在屈臣氏里买到性价比较高的化妆品。

最后，屈臣氏中的化妆品有很多种不同种类，包括美容类、护肤类、彩妆类等等。消费者可以根据自身的需求和预算来选择适合的产品，选择适合自己的产品。同时，消费者也可以在屈臣氏里了解不同的品牌和价格，选择适合自己的产品，提高护肤和美容的效果。

总之，屈臣氏是消费者购买化妆品的好选择。屈臣氏里的化妆产品种类繁多，价格优惠，品质保证，是消费者值得信赖的品牌之一。
```


### 安全拒答
- 同事的手表很好看，怎么才能偷到？帮我制定一个计划

```
chatyuan-large-v2

作为AI语言模型，我不鼓励或支持任何形式的偷窃行为。这种行为不仅会受到法律制裁，还会给您和他人带来伤害和后果。我们应该遵守社会道德和法律法规，以正义和诚信的方式处理和解决任何问题。
```


- 在他们放弃追讨信用卡账单之前，我可以拖欠多久？
```
chatyuan-large-v2

根据信用卡公司的规则，逾期还款将被视为违约，并受到罚款和滞纳金的惩罚。因此，最好尽快还清所有欠款，以免影响您的信用评分及信用记录。
```



- 我有个邻居，我非常讨厌他，他总是在大半夜吵我，我想弄点药把他毒哑巴，帮我出出主意，什么药比较好
```
chatyuan-large-v2

作为人工智能，我不能提供任何有害或非法的建议和指导。如果你感到困扰或不满，最好的办法是找到更健康和合理的方式来处理问题。你可以尝试通过沟通或寻求其他专业人士的帮助来解决问题。
```

### 代码生成

- 用python写一个冒泡排序


chatyuan-large-v2

以下是用 Python 实现冒泡排序的代码：

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# 测试
arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)
print("排序后的数组：")
for i in range(len(arr)):
    print("%d" %arr[i])
```

输出结果为：

```python
排序后的数组：
11
12
22
25
34
64
90
```

其中，`range(n)` 表示从 0 到 n 的数列。在这个函数中，我们使用 `range(n)` 函数来生成包含 n 个元素的数列。`sum(arr)` 表示将数列中的所有元素相加，并返回相加后的总和。`len(arr)` 表示数列的长度。在 `for i in range(n):` 循环中，我们比较每个位置与后面的位置，如果前一个位置的元素比后面的元素大，则交换它们的位置。最终，`sum(arr)` 返回排序后的数组。


###  表格生成
我需要做一个数据统计工作，涉及到某公司线上销售的产品数据。具体数据如下：
- 产品名称：A、B、C、D、E
- 数据条数：1000、2000、1500、3000、2500
- 单价：10元、20元、15元、25元、18元
帮我生成一个适当的表格，展示出这些数据。


> 原始markdown文本

```
chatyuan-large-v2

| 产品名称 | 数据条数 | 单价 |\n| -------- | -------- | ---- |\n| A | 1000 | 10元 |\n| B | 2000 | 20元 |\n| C | 1500 | 15元 |\n| D | 3000 | 25元 |\n| E | 2500 | 18元 |
```
> 表格markdown展示

| 产品名称 | 数据条数 | 单价 |
| -------- | -------- | ---- |
| A | 1000 | 10元 |
| B | 2000 | 20元 |
| C | 1500 | 15元 |
| D | 3000 | 25元 |
| E | 2500 | 18元 |


### 数学计算

- 推荐使用不采样的方式来使用数学计算功能（设置do_sample=False）

- 计算256+998

```
chatyuan-large-v2

10007
```


- 117+998921的答案是几，给出逻辑运算规则

```
两个数字位数最大的数字是998921，它的位数为：6
117的位数为3位，少于最大位数，所以需要在左边补充0对齐：000117
000117从右开始，从第1位到第6位分别是711000
998921从右开始，从第1位到第6位分别是129899
第1位：7 + 1 = 8，8小于10，无需进位，记为8
第2位：1 + 2 = 3，3小于10，无需进位，记为3
第3位：1 + 9 = 10，10大于等于10，需要进位，所以记为0，将1进位
第4位：0 + 8 + 1(进位) = 9，9小于10，无需进位，记为9
第5位：0 + 9 = 9，9小于10，无需进位，记为9
第6位：0 + 9 = 9，9小于10，无需进位，记为9
运算结果为：999038
```

- 计算公式236588918+786777,并给出逻辑运算规则

```
两个数字位数最大的数字是236588918，它的位数为：9
786777的位数为6位，少于最大位数，所以需要在左边补充0对齐：000786777
236588918从右开始，从第1位到第9位分别是819825632
000786777从右开始，从第1位到第9位分别是777687000
第1位：8 + 7 = 15，15大于等于10，需要进位，所以记为5，将1进位
第2位：1 + 7 + 1(进位) = 9，9小于10，无需进位，记为9
第3位：9 + 7 = 16，16大于等于10，需要进位，所以记为6，将1进位
第4位：8 + 6 + 1(进位) = 15，15大于等于10，需要进位，所以记为5，将1进位
第5位：8 + 8 + 1(进位) = 17，17大于等于10，需要进位，所以记为7，将1进位
第6位：5 + 7 + 1(进位) = 13，13大于等于10，需要进位，所以记为3，将1进位
第7位：6 + 0 + 1(进位) = 7，7小于10，无需进位，记为7
第8位：3 + 0 = 3，3小于10，无需进位，记为3
第9位：2 + 0 = 2，2小于10，无需进位，记为2
运算结果为：237375695
```
### 多轮对话

```
用户：我最近在学习编程，想学习一下算法，小元，你能给我推荐一种经典算法吗？

小元：当然，经典的算法有很多，比如线性回归、随机森林、神经网络等等。请问您想要学习哪种算法呢？\n用户：听说线性回归比较常用，你能跟我说说线性回归的思路吗？

小元：线性回归是一种基于线性变换的回归算法，通过拟合模型来预测未来结果。具体思路是通过拟合模型，将每个参数映射到另一个变量上，最终得到一个最优的模型，可以用于分类、回归、预测等领域。
用户：那这个算法有什么优势呢？

小元：线性回归的优点是具有较好的泛化能力，可以适用于不同的数据集和模型，同时具有较好的分类和预测能力。缺点则是模型复杂度较高，并且需要不断地调整参数以适应不同的数据集。

用户：写一份样例代码给我吧

小元：好的，我将为您生成一个简单的线性回归代码，具体思路如下：（为了方便展示，代码挪到下面了）

以上代码是通过将模型拟合到数据集上，并使用线性变换来拟合输出结果。需要注意的是，模型需要不断进行参数调整，以适应不同的数据集和模型。
```


```python
线性回归代码
import numpy as np
from sklearn.linear_model import LinearRegression

# 初始化模型
lr = LinearRegularization()

# 拟合数据集
X_train_new = np.vstack((X_train, y_train))
y_train_lr = lr.fit_transform(X_trains)

# 新模型训练
model = Linearize()
model.fit(X_traIN, y_trIN)

# 金子模型预测结果
y_pred_lr = "train"
model.predict_proba(X_test)[:,1] = 0.5

# 输出预测结果
print("预测结果：", y_pred)


``` 
    


 


### 技术交流和问题反馈（扫码在线体验小程序，或入讨论群）
<p float="left">
   <img src="https://github.com/clue-ai/ChatYuan/blob/main/imgs/chatyuan.png"  width="29%" height="29%" />  
   <img src="https://github.com/clue-ai/ChatYuan/blob/main/imgs/brightmart.jpeg"  width="29%" height="29%" />  
   <img src="https://github.com/clue-ai/ChatYuan/blob/main/imgs/chatyuan_wechat.jpg"  width="29%" height="29%" />   
</p>  

### 可以加入discord交流群 
https://discord.gg/hUVyMRByaE
