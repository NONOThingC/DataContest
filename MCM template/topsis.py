def tolerance(data, *args):
    x_min = args[0][0]
    x_max = args[0][1]
    x_minimum = args[1][0]
    x_maximum = args[1][1]
    def normalization(data):
        if data >= x_min and data <= x_max:
            data = 1
        elif data <= x_minimum or data >= x_maximum:
            data = 0
        elif data > x_max and data < x_maximum:
            data = 1 - (data - x_max) / (x_maximum - x_max)
        elif data < x_min and data > x_minimum:
            data = 1 - (x_min - data) / (x_min - x_minimum)
        return data
    return list(map(normalization,data))

if __name__ == '__main__':
    # 示例：对 [5,6,7,10,12] 这组数据，在最佳稳定区间 [5, 6]，最大容忍区间 [2, 12] 上做变换
    tolerance([5, 6, 7, 10, 12], [5, 6], [2, 12])
def topsis(data, w='', index=0, normalize=False):
    import pandas as pd
    import numpy as np
    # 同向化
    if type(index) != int:
        for i in index:
            data.iloc[:, i - 1] = 1 / data.iloc[:, i - 1]
            data.columns = list(data.columns[:i - 1]) + [list(data.columns)[i - 1] + '(反)'] + list(data.columns[i:])
    # Z分数变换规范化，默认不使用
    if normalize:
        data = (data - data.mean()) / data.std()
        print('规范化矩阵:\n', data)
    # 归一化
    data = data / np.sqrt((data ** 2).sum())
    print('归一化矩阵:\n', data)
    # 最优最劣方案
    Z_positive = data.max()
    Z_negative = data.min()
    print('\n最优方案为\n', Z_positive)
    print('\n最劣方案为\n', Z_negative)
    # 距离
    if w == '':
        w = 1
    D_positive = np.sqrt(((data - Z_positive) ** 2 * w).sum(axis=1).values)
    D_negative = np.sqrt(((data - Z_negative) ** 2 * w).sum(axis=1).values)

    # 贴近程度
    C = D_negative / (D_negative + D_positive)
    out = pd.DataFrame({'正理想解': D_positive, '负理想解': D_negative, '最终得分': C}, index=data.index)
    out['排序'] = out.rank(ascending=False)['最终得分']

    print(out)
    return out, Z_positive, Z_negative, data

import pandas as pd
import numpy as np
#普通办法实例
data = pd.DataFrame(
        {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
         '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])
data['生师比'] = tolerance(data['生师比'], [5, 6], [2, 12])   # 师生比数据为区间型指标
out = topsis(data, index = [4],w=np.array([0.2, 0.3, 0.4, 0.1]))    # 设置权系数
#熵权法实例
data = pd.DataFrame(
        {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
         '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])
# 归一化
data = data / data.sum()
print('归一化矩阵：\n', data)
# 计算各指标熵值
k = 1 / np.log(len(data))
e = (-k * data * np.log(data)).sum()
print('各指标熵值：\n', e)
# 计算权系数
h = (1 - e) / (1 - e).sum()
print('各指标权系数：\n', h)


# 导入第三方模块
import numpy as np
import matplotlib.pyplot as plt

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 使用ggplot的绘图风格
plt.style.use('ggplot')

# 构造数据
values = [3.2,2.1,3.5,2.8,3]
values2 = [4,4.1,4.5,4,4.1]
feature = ['个人能力','QC知识','解决问题能力','服务质量意识','团队精神']

N = len(values)
# 设置雷达图的角度，用于平分切开一个圆面
angles=np.linspace(0, 2*np.pi, N, endpoint=False)
# 为了使雷达图一圈封闭起来，需要下面的步骤
values=np.concatenate((values,[values[0]]))
values2=np.concatenate((values2,[values2[0]]))
angles=np.concatenate((angles,[angles[0]]))

# 绘图
fig=plt.figure()
ax = fig.add_subplot(111, polar=True)
# 绘制折线图
ax.plot(angles, values, 'o-', linewidth=2, label = '活动前')
# 填充颜色
ax.fill(angles, values, alpha=0.25)
# 绘制第二条折线图
ax.plot(angles, values2, 'o-', linewidth=2, label = '活动后')
ax.fill(angles, values2, alpha=0.25)

# 添加每个特征的标签
ax.set_thetagrids(angles * 180/np.pi, feature)
# 设置雷达图的范围
ax.set_ylim(0,5)
# 添加标题
plt.title('活动前后员工状态表现')

# 添加网格线
ax.grid(True)
# 设置图例
plt.legend(loc = 'best')
# 显示图形
plt.show()

"""
# 导入第三方模块
import pygal

# 调用Radar这个类，并设置雷达图的填充，及数据范围
radar_chart = pygal.Radar(fill = True, range=(0,5))
# 添加雷达图的标题
radar_chart.title = '活动前后员工状态表现'
# 添加雷达图各顶点的含义
radar_chart.x_labels = ['个人能力','QC知识','解决问题能力','服务质量意识','团队精神']

# 绘制两条雷达图区域
radar_chart.add('活动前', [3.2,2.1,3.5,2.8,3])
radar_chart.add('活动后', [4,4.1,4.5,4,4.1])

# 保存图像
radar_chart.render_to_file('radar_chart.svg')
"""