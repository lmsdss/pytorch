import paramiko
import os

client = paramiko.SSHClient()  # 获取SSHClient实例
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.31.53", username="bb", password="bb")  # 连接SSH服务端
transport = client.get_transport()  # 获取Transport实例

# 创建sftp对象，SFTPClient是定义怎么传输文件、怎么交互文件
sftp = paramiko.SFTPClient.from_transport(transport)

# 将本地 api.py 上传至服务器 /www/test.py。文件上传并重命名为test.py
sftp.put("/home/ustc-swf/cuda-test.py", os.path.join("/home/bb", 'cuda-test.py'))

# 将服务器 /www/test.py 下载到本地 aaa.py。文件下载并重命名为aaa.py
# sftp.get("/www/test.py", "E:/aaa.py")

# 关闭连接
client.close()