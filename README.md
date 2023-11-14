# PS

- 最好使用`Python3.8`
- 不要忘记`~/.mujoco/mujoco210/`
- 使用`pip`安装完依赖包后还要手动修改一下

```sh
cd .venv/lib/python3.8/site-packages/
ln -s mujoco_py mujoco  # 创建软链接，解决“import mujoco”报错问题
cd -
```

- 运行时需要设置环境变量，也可在全局设置

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
# 如果你使用显卡，还要运行以下命令
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
