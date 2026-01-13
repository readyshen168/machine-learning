# k 近邻算法

## import kNN 后再目录中会出现**pycache**,包含文件 kNN.cpython-313.pyc

- kNN：对应你的模块名

- cpython：表示使用的是 CPython 解释器

- XX：Python 版本号（如 38 表示 Python 3.8）

- .pyc：编译后的字节码文件

### Python 解释器为了提高模块加载速度

首次导入模块时，会将源代码编译为字节码

将字节码缓存到 **pycache**目录

下次导入时直接加载字节码，跳过编译步骤

### 这些文件的作用

加速加载 ​​：避免每次重复编译源代码

​​ 一致性检查 ​​：通过比较源文件和字节码的时间戳判断是否需要重新编译

​​ 跨运行保持 ​​：多个 Python 进程可以共享相同的字节码缓存

### 可以安全删除 ​​：Python 会在需要时重新生成

`rm -rf __pycache__/`

### 最佳实践

将 **pycache**加入.gitignore​​：

```vim
__pycache__/
*.py[cod]
```
