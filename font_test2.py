import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']
#plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
  
t =  1
y =  2
plt.plot(t, y) 
plt.title(u'这里写的是中文')  
plt.xlabel(u'X坐标')  
plt.ylabel(u'Y坐标')  
plt.show()
