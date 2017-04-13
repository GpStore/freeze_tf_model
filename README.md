# freeze_tf_model

一个将 tensorflow 模型静态化的工具，方便微服务加载

Tensorflow 的 tf.train.Saver 保存的模型 checkpoint 无法直接被 serving 代码调用，这个工具从 checkpoint 导入模型，将模型中的所有参数常量化，
然后将模型保存在单一文件中方便调用。

**使用方法**

```
./freeze_model.py --model_dir <checkpoint 保存的目录> --output_node_names <需要输出的节点名>
```

注意：

1. checkpoint 保存的目录最后不要有 "/"
2. 需要输出的节点名可以有多个，用逗号分隔，如果你用 golang 实现 serving 服务，通常是 session.Run 中的 output 节点的名字，如 
[这段代码](https://github.com/huichen/gotalk/blob/master/beam_search.go) 中包含了三个输出节点，"lstm/initial_state", "softmax", "lstm/state"。
3. 输出的 frozen_model.pb 保存在 --model_dir 中
