---
title: numpy的argmax的具体使用
subtitle: numpy.argmax的通俗理解
summary: numpy.argmax的通俗理解
authors:
- admin
tags:
- numpy
- 工具
categories:
- python
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  placement: 1
  caption: ''
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---











## numpy.argmax




假定现在有一个数组a = [3, 1, 2, 4, 6, 1]现在要算数组a中最大数的索引是多少.这个问题对于刚学编程的同学就能解决.最直接的思路,先假定第0个数最大,然后拿这个和后面的数比,找到大的就更新索引.代码如下


```python
a = [3, 1, 2, 4, 6, 1]
maxindex = 0
i = 0
for tmp in a:
  if tmp > a[maxindex]:
    maxindex = i
  i += 1
print(maxindex)


```

    4
    

还是从一维数组出发.看下面的例子.


```python
import numpy as np
a = np.array([3, 1, 2, 4, 6, 1])
print(np.argmax(a))

```

    4
    

argmax返回的是最大数的索引.argmax有一个参数axis,默认是0,表示第几维的最大值.看二维的情况.


```python
import numpy as np
a = np.array([[1, 5, 5, 2],
       [9, 6, 2, 8],
       [3, 7, 9, 1]])
print(np.argmax(a, axis=0))
```

    [1 2 2 1]
    

为了描述方便,a就表示这个二维数组.np.argmax(a, axis=0)的含义是a[0][j],a[1][j],a[2][j](j=0,1,2,3)中最大值的索引.从a[0][j]开始,最大值索引最初为(0,0,0,0),拿a[0][j]和a[1][j]作比较,9大于1,6大于5,8大于2,所以最大值索引由(0,0,0,0)更新为(1,1,0,1),再和a[2][j]作比较,7大于6,9大于5所以更新为(1,2,2,1).再分析下面的输出.


```python
import numpy as np
a = np.array([[1, 5, 5, 2],
       [9, 6, 2, 8],
       [3, 7, 9, 1]])
print(np.argmax(a, axis=1))
```

    [1 0 2]
    

np.argmax(a, axis=1)的含义是a[i][0],a[i][1],a[i][2],a[i][3](i=0,1,2)中最大值的索引.从a[i][0]开始,a[i][0]对应的索引为(0,0,0),先假定它就是最大值索引(思路和上节简单例子完全一致)拿a[i][0]和a[i][1]作比较,5大于1,7大于3所以最大值索引由(0,0,0)更新为(1,0,1),再和a[i][2]作比较,9大于7,更新为(1,0,2),再和a[i][3]作比较,不用更新,最终值为(1,0,2)
再看三维的情况.

## 再尝试一下简单的三维例子


```python
import numpy as np
a = np.array([
       [
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]
       ],

       [
         [4, 4, 4, 4],
         [5, 5, 5, 5],
         [6, 6, 6, 6]
       ]
      ])
print(np.argmax(a, axis=0))



```

    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]
    

我们把第三行换成[7,7,7]再试试，发现


```python
a = np.array([
       [
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [7, 7, 7, 7]
       ],

       [
         [4, 4, 4, 4],
         [5, 5, 5, 5],
         [6, 6, 6, 6]
       ]
      ])
print(np.argmax(a, axis=0))
```

    [[1 1 1 1]
     [1 1 1 1]
     [0 0 0 0]]
    

再把第一行改成[1,1,1,9]


```python
a = np.array([
       [
         [1, 1, 1, 9],
         [2, 2, 2, 2],
         [7, 7, 7, 7]
       ],

       [
         [4, 4, 4, 4],
         [5, 5, 5, 5],
         [6, 6, 6, 6]
       ]
      ])
print(np.argmax(a, axis=0))
```

    [[1 1 1 0]
     [1 1 1 1]
     [0 0 0 0]]
    

因此，np.argmax(a, axis=0)类似同一个位置哪个通道的值最大

那么,np.argmax(a, axis=1)呢


```python
a = np.array([
       [
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]
       ],

       [
         [4, 4, 4, 4],
         [5, 5, 5, 5],
         [6, 6, 6, 6]
       ]
      ])
print(np.argmax(a, axis=1))
```

    [[2 2 2 2]
     [2 2 2 2]]
    

我们把第二行换成[7,7,7,7]再试试，发现


```python
a = np.array([
       [
         [1, 1, 1, 1],
         [7, 7, 7, 7],
         [3, 3, 3, 3]
       ],

       [
         [4, 4, 4, 4],
         [5, 5, 5, 5],
         [6, 6, 6, 6]
       ]
      ])
print(np.argmax(a, axis=1))
```

    [[1 1 1 1]
     [2 2 2 2]]
    

可以推断，np.argmax(a, axis=1)类似于告诉我们每个通道中最大的一行是哪行


