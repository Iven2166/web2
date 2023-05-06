---
title: "python"
date: 2022-11-01
toc: true
categories:
  - python
classes: wide
words_per_minute: 10
---

# 常见问题

### list扩容机制

机制：
- 4个起步，之后的扩容是 2倍扩容

可能发生的问题和解决方案
- 过大：可以自定义list，

- 参考
  - [http://blog.itpub.net/69923331/viewspace-2694185/](http://blog.itpub.net/69923331/viewspace-2694185/)

### 使用args和*kwargs的含义

- 当我们不知道向函数传递多少参数时，比如我们向传递一个列表或元组，我们就使用*args
`
def func(*args):
    for i in args:
        print(i)  
func(3,2,1,4,7)
`
- 在我们不知道该传递多少关键字参数时，使用**kwargs来收集关键字参数。
`
def func(**kwargs):
    for i in kwargs:
        print(i,kwargs[i])
func(a=1,b=2,c=7)
`

### python 闭包

- 函数有嵌套时，里面的方程即使不定义入参，也可以拿到外层函数的变量。因此，类似于全局变量的作用。
- 外部函数执行后，即便我们再 `del outerfunc` 执行 `innerfunc()`依旧会给出"记忆"到的变量

好处：
- 不需要用到全局变量，起到数据隐藏的效果
- 嵌套函数，不需要另外写一个类

参考：
- [https://data-flair.training/blogs/python-closure/](https://data-flair.training/blogs/python-closure/)

### 当Python退出时，为什么不清除所有分配的内存？

当Python退出时，尤其是那些对其他对象具有循环引用的Python模块或者从全局名称空间引用的对象并没有被解除分配或释放。

无法解除分配C库保留的那些内存部分。

退出时，由于拥有自己的高效清理机制，Python会尝试取消分配/销毁其他所有对象。