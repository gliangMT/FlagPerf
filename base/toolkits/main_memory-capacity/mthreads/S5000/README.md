# 参评AI芯片信息

* 厂商：MThreads
* 产品名称：S5000
* 产品型号：MTT S5000
* TDP：/

# 所用服务器配置

* 服务器数量：1
* 单服务器内使用卡数：2
* 服务器型号：/
* 操作系统版本：Ubuntu 22.04.4 LTS
* 操作系统内核：Linux 5.15.0-105-generic
* CPU：/
* docker版本：24.0.7
* 内存：2TiB
* 服务器间AI芯片直连规格及带宽：此评测样例无需服务器间通信

# 评测结果

## 核心评测结果

| 评测项  | 主存储容量测试值          | 主存储容量标定值 | 测试标定比例 |
| ---- | ----------------- | -------- | ------ |
| 评测结果 | / | / | /  |

## 能耗监控结果

此评测样例中无意义

## 其他重要监控结果

| 监控项  | 系统平均CPU占用 | 系统平均内存占用 |
| ---- | --------- | -------- |
| 监控结果 | /   | /   |

# 厂商测试工具原理说明

通过按照一定规则不断尝试申请主存储（例如显存）来评测主存储容量

1. 初始化某个INITSIZE
2. 不断尝试musaMalloc INITSIZE大小的主存储，直到无法申请
3. 减小INITSIZE为当前的二分之一，重复执行第2步
4. 重复执行第3步，直到INITSIZE为1MiB

上述评测过程可以确保在评测结束时，已无法申请任何1MiB的主存储，以此评测主存储容量。