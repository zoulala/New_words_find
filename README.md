# New_words_find
新词发现，信息熵，左右互信息

我们常使用jieba分词来作为语言处理的第一到工序，但是对行业领域的专业词、新词却很难区分，该程序通过统计行业领域的语料数据，用信息熵技术实现新词的发现。

# input
语料文档txt
```
万灵石4级哪里可以获取
天下第一擂是干什么的呢。。。
雄才伟略
宝宝丢了，伤心啊
其他点卡可以充值不？
苏州打造台
小票获得什么
游戏视野这么拉远
黄龙洞在哪里
宝宝套怎么升星？
比武大会什么时候开始
制作装备的过程，需要多长时间
3万评分时什么级别的擂台
.
.
.
```

# output
语料中的词，并依据次数排序
```
怎么 2836108
是什么 1551873
账号 896048
在哪 825500
如何 784982
多少 774153
珍兽 661080
哪里 613245
可以 598351
任务 578104
技能 540769
装备 529157
手机 511519
升级 494052
5级 488499
神器 461028
钓鱼 425015
宝石 394782
密码 377159
称号 374558
怎么获得 366338
在哪里 317517
宝宝 301391
属性 301280
武器 300026
.
.
.
```
