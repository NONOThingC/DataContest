# JavaScript 注释（同C语言）
- 单行注释以 // 开头
- 多行注释以 /* 开始，以 */ 结尾。

# JavaScript 变量
与代数一样，JavaScript 变量可用于存放值（比如 x=2）和表达式（比如 z=x+y）。

变量必须以字母开头
变量也能以 $ 和 _ 符号开头（不过我们不推荐这么做）
变量名称对大小写敏感（y 和 Y 是不同的变量）
提示：JavaScript 语句和 JavaScript 变量都对大小写敏感。
## 声明（创建） JavaScript 变量
在 JavaScript 中创建变量通常称为“声明”变量。

我们使用 var 关键词来声明变量。
```
var carname;
```
Value = undefined
在计算机程序中，经常会声明无值的变量。未使用值来声明的变量，其值实际上是 undefined。

在执行过以下语句后，变量 carname 的值将是 undefined：
```
var carname;
```
## Undefined 和 Null
Undefined 这个值表示变量不含有值。
可以通过将变量的值设置为 null 来清空变量。
## 重新声明 JavaScript 变量
如果重新声明 JavaScript 变量，该变量的值不会丢失：

在以下两条语句执行后，变量 carname 的值依然是 "Volvo"：
```
var carname="Volvo";
var carname;
```
## 声明变量类型
当您声明新变量时，可以使用关键词 "new" 来声明其类型：
```
var carname=new String;
var x=      new Number;
var y=      new Boolean;
var cars=   new Array;
var person= new Object;
```
JavaScript 变量均为对象。当您声明一个变量时，就创建了一个新的对象。

## 局部 JavaScript 变量
在 JavaScript 函数内部声明的变量（使用 var）是局部变量，所以只能在函数内部访问它。（该变量的作用域是局部的）。

您可以在不同的函数中使用名称相同的局部变量，因为只有声明过该变量的函数才能识别出该变量。

只要函数运行完毕，本地变量就会被删除。

## 全局 JavaScript 变量
在函数外声明的变量是全局变量，网页上的所有脚本和函数都能访问它。

## JavaScript 变量的生存期
JavaScript 变量的生命期从它们被声明的时间开始。

局部变量会在函数运行以后被删除。

全局变量会在页面关闭后被删除。

# JavaScript 数据类型
1. JavaScript 变量还能保存其他数据类型，比如文本值 (name="Bill Gates")。
在 JavaScript 中，类似 "Bill Gates" 这样一条文本被称为字符串。当您向变量分配文本值时，应该用双引号或单引号包围这个值。
当您向变量赋的值是数值时，不要使用引号。如果您用引号包围数值，该值会被作为文本来处理。
2. JavaScript 布尔
布尔（逻辑）只能有两个值：true 或 false。
3. JavaScript 拥有动态类型
JavaScript 拥有动态类型。这意味着相同的变量可用作不同的类型：
```
实例
var x                // x 为 undefined
var x = 6;           // x 为数字
var x = "Bill";      // x 为字符串
```
4. JavaScript 数组
下面的代码创建名为 cars 的数组：
```
var cars=new Array();
cars[0]="Audi";
cars[1]="BMW";
cars[2]="Volvo";
或者 (condensed array):

var cars=new Array("Audi","BMW","Volvo");
或者 (literal array):

实例
var cars=["Audi","BMW","Volvo"];
```
  数组下标是基于零的，所以第一个项目是 [0]，第二个是 [1]，以此类推。

5. JavaScript 对象
对象由花括号分隔。在括号内部，对象的属性以名称和值对的形式 (name : value) 来定义。属性由逗号分隔：
```
var person={firstname:"Bill", lastname:"Gates", id:5566};
```
上面例子中的对象 (person) 有三个属性：firstname、lastname 以及 id。

空格和折行无关紧要。声明可横跨多行：
```
var person={
firstname : "Bill",
lastname  : "Gates",
id        :  5566
};
```
对象属性有两种寻址方式：

实例
```
name=person.lastname;
name=person["lastname"];
```
# JavaScript 函数语法
函数就是包裹在花括号中的代码块，前面使用了关键词 function：

function functionname()
{
这里是要执行的代码
}
当调用该函数时，会执行函数内的代码。

可以在某事件发生时直接调用函数（比如当用户点击按钮时），并且可由 JavaScript 在任何位置进行调用。

提示：JavaScript 对大小写敏感。关键词 function 必须是小写的，并且必须以与函数名称相同的大小写来调用函数。

## 带有返回值的函数
有时，我们会希望函数将值返回调用它的地方。

通过使用 return 语句就可以实现。

在使用 return 语句时，函数会停止执行，并返回指定的值。

语法
```
function myFunction()
{
var x=5;
return x;
}
```
上面的函数会返回值 5。

注释：整个 JavaScript 并不会停止执行，仅仅是函数。JavaScript 将继续执行代码，从调用函数的地方。

函数调用将被返回值取代：

var myVar=myFunction();
myVar 变量的值是 5，也就是函数 "myFunction()" 所返回的值。

即使不把它保存为变量，您也可以使用返回值：
```
document.getElementById("demo").innerHTML=myFunction();
```
"demo" 元素的 innerHTML 将成为 5，也就是函数 "myFunction()" 所返回的值。

您可以使返回值基于传递到函数中的参数：

实例
计算两个数字的乘积，并返回结果：
```
function myFunction(a,b)
{
return a*b;
}

document.getElementById("demo").innerHTML=myFunction(4,3);
```
"demo" 元素的 innerHTML 将是：
12
## JSP常用函数
if,while,for,switch语法基本同C语言，该变的地方变即可。
try,catch,throw同C++.

# JavaScript 表单验证
## JavaScript 表单验证
JavaScript 可用来在数据被送往服务器前对 HTML 表单中的这些输入数据进行验证。

被 JavaScript 验证的这些典型的表单数据有：

用户是否已填写表单中的必填项目？
用户输入的邮件地址是否合法？
用户是否已输入合法的日期？
用户是否在数据域 (numeric field) 中输入了文本？
### 必填（或必选）项目
下面的函数用来检查用户是否已填写表单中的必填（或必选）项目。假如必填或必选项为空，那么警告框会弹出，并且函数的返回值为 false，否则函数的返回值则为 true（意味着数据没有问题）：
```
function validate_required(field,alerttxt)
{
with (field)
{
if (value==null||value=="")
  {alert(alerttxt);return false}
else {return true}
}
}
```
下面是连同 HTML 表单的代码：
```
<html>
<head>
<script type="text/javascript">

function validate_required(field,alerttxt)
{
with (field)
  {
  if (value==null||value=="")
    {alert(alerttxt);return false}
  else {return true}
  }
}

function validate_form(thisform)
{
with (thisform)
  {
  if (validate_required(email,"Email must be filled out!")==false)
    {email.focus();return false}
  }
}
</script>
</head>

<body>
<form action="submitpage.htm" onsubmit="return validate_form(this)" method="post">
Email: <input type="text" name="email" size="30">
<input type="submit" value="Submit"> 
</form>
</body>

</html>
```
### E-mail 验证
下面的函数检查输入的数据是否符合电子邮件地址的基本语法。

意思就是说，输入的数据必须包含 @ 符号和点号(.)。同时，@ 不可以是邮件地址的首字符，并且 @ 之后需有至少一个点号：
```
function validate_email(field,alerttxt)
{
with (field)
{
apos=value.indexOf("@")
dotpos=value.lastIndexOf(".")
if (apos<1||dotpos-apos<2) 
  {alert(alerttxt);return false}
else {return true}
}
}
```
下面是连同 HTML 表单的完整代码：
```
<html>
<head>
<script type="text/javascript">
function validate_email(field,alerttxt)
{
with (field)
{
apos=value.indexOf("@")
dotpos=value.lastIndexOf(".")
if (apos<1||dotpos-apos<2) 
  {alert(alerttxt);return false}
else {return true}
}
}

function validate_form(thisform)
{
with (thisform)
{
if (validate_email(email,"Not a valid e-mail address!")==false)
  {email.focus();return false}
}
}
</script>
</head>

<body>
<form action="submitpage.htm"onsubmit="return validate_form(this);" method="post">
Email: <input type="text" name="email" size="30">
<input type="submit" value="Submit"> 
</form>
</body>

</html>
```

# JavaScript HTML DOM
HTML DOM （文档对象模型）
当网页被加载时，浏览器会创建页面的文档对象模型（Document Object Model）。

HTML DOM 模型被构造为对象的树。

HTML DOM 树示意图如下：
![1.jpg](ct_htmltree.gif)
通过可编程的对象模型，JavaScript 获得了足够的能力来创建动态的 HTML。
- JavaScript 能够改变页面中的所有 HTML 元素
- JavaScript 能够改变页面中的所有 HTML 属性
- JavaScript 能够改变页面中的所有 CSS 样式
- JavaScript 能够对页面中的所有事件做出反应

## 改变 HTML 输出流
JavaScript 能够创建动态的 HTML 内容：

今天的日期是： Tue Jan 22 2019 22:59:39 GMT+0800 (中国标准时间)

在 JavaScript 中，document.write() 可用于直接向 HTML 输出流写内容。
```
实例
<!DOCTYPE html>
<html>
<body>

<script>
document.write(Date());
</script>

</body>
</html>
```

提示：绝不要使用在文档加载之后使用 document.write()。这会覆盖该文档。

## 改变 HTML 内容
修改 HTML 内容的最简单的方法时使用 innerHTML 属性。

如需改变 HTML 元素的内容，请使用这个语法：
```
document.getElementById(id).innerHTML=new HTML
```
实例
本例改变了 <p> 元素的内容：
```
<html>
<body>

<p id="p1">Hello World!</p>

<script>
document.getElementById("p1").innerHTML="New text!";
</script>

</body>
</html>
```

实例

本例改变了 <h1> 元素的内容：
```
<!DOCTYPE html>
<html>
<body>

<h1 id="header">Old Header</h1>

<script>
var element=document.getElementById("header");
element.innerHTML="New Header";
</script>

</body>
</html>
```
例子解释：

上面的 HTML 文档含有 id="header" 的 <h1> 元素
我们使用 HTML DOM 来获得 id="header" 的元素
JavaScript 更改此元素的内容 (innerHTML)
改变 HTML 属性
如需改变 HTML 元素的属性，请使用这个语法：

document.getElementById(id).attribute=new value
实例
本例改变了 <img> 元素的 src 属性：
```
<!DOCTYPE html>
<html>
<body>

<img id="image" src="smiley.gif">

<script>
document.getElementById("image").src="landscape.jpg";
</script>

</body>
</html>
```

例子解释：

上面的 HTML 文档含有 id="image" 的 <img> 元素
我们使用 HTML DOM 来获得 id="image" 的元素
JavaScript 更改此元素的属性（把 "smiley.gif" 改为 "landscape.jpg"）

HTML DOM 允许 JavaScript 改变 HTML 元素的样式。

## 改变 HTML 样式
如需改变 HTML 元素的样式，请使用这个语法：
```
document.getElementById(id).style.property=new style
```
例子 1
下面的例子会改变 <p> 元素的样式：
```
<p id="p2">Hello World!</p>
<script>
document.getElementById("p2").style.color="blue";
</script>
```
# 易遗漏点

- JavaScript 对大小写是敏感的。
当编写 JavaScript 语句时，请留意是否关闭大小写切换键。
函数 getElementById 与 getElementbyID 是不同的。
同样，变量 myVariable 与 MyVariable 也是不同的。
- 对代码行进行折行
您可以在文本字符串中使用反斜杠对代码行进行换行。
- 