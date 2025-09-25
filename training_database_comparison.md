# 训练数据库设计对比：文档 vs 实际实现

## 表名对比

| 数据库表 | 文档中的表名 | 实际代码中的表名 | 状态 |
|----------|-------------|-----------------|------|
| 训练任务表 | `training_tasks_v2` | `training_tasks` | ❌ 不一致 |
| 数据集信息表 | `dataset_info_v2` | `dataset_info` | ❌ 不一致 |

## training_tasks 表字段对比

| 字段名 | 文档 | 实际实现 | 一致性 | 备注 |
|--------|------|----------|--------|------|
| task_id | ✅ | ✅ | ✅ | 主键，一致 |
| task_name | ✅ | ✅ | ✅ | 可选字段 |
| description | ✅ | ✅ | ✅ | 可选字段 |
| train_type | ✅ | ✅ | ✅ | embedding/reranker |
| dataset_name_or_path | ✅ | ✅ | ✅ | 数据集路径 |
| **HF_subset** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |
| output_dir | ✅ | ✅ | ✅ | 输出目录 |
| model_name_or_path | ✅ | ✅ | ✅ | 可选字段 |
| embedding_dim | ✅ | ✅ | ✅ | 可选字段 |
| **device** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |
| status | ✅ | ✅ | ✅ | 训练状态 |
| progress | ✅ | ✅ | ✅ | 0.0-100.0 |
| created_at | ✅ | ✅ | ✅ | 创建时间 |
| **updated_at** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |
| started_at | ✅ | ✅ | ✅ | 开始时间 |
| completed_at | ✅ | ✅ | ✅ | 完成时间 |
| **duration_seconds** | ✅ | ❌ 缺失 | ❌ | 文档中有但实际没有 |
| final_model_path | ✅ | ✅ | ✅ | 最终模型路径 |
| error_message | ✅ | ✅ | ✅ | 错误信息 |
| **loss_data** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |
| training_params | ✅ | ✅ | ✅ | JSON格式 |
| **service_instance_id** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |
| **process_pid** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |
| **process_status** | ❌ 缺失 | ✅ | ❌ | 实际实现中有此字段 |

## dataset_info 表字段对比

| 字段名 | 文档 | 实际实现 | 一致性 | 备注 |
|--------|------|----------|--------|------|
| id | ✅ | ✅ | ✅ | 主键 |
| task_id | ✅ | ✅ | ✅ | 外键 |
| data_source_id | ✅ | ✅ | ✅ | 数据源ID |
| dataset_name | ✅ | ✅ | ✅ | 基础名称 |
| dataset_base_name | ✅ | ✅ | ✅ | 向后兼容 |
| HF_subset | ✅ | ✅ | ✅ | HuggingFace子配置 |
| dataset_path | ✅ | ✅ | ✅ | 路径或名称 |
| dataset_type | ✅ | ✅ | ✅ | 数据集类型 |
| split_type | ✅ | ✅ | ✅ | train/eval/test |
| dataset_status | ✅ | ✅ | ✅ | pending/loaded/failed |
| evaluation_status | ✅ | ✅ | ✅ | 评估状态 |
| error_message | ✅ | ✅ | ✅ | 错误信息 |
| total_samples | ✅ | ✅ | ✅ | 样本总数 |
| configured_sample_size | ✅ | ✅ | ✅ | 实际使用样本数 |
| target_column | ✅ | ✅ | ✅ | 目标列名 |
| label_type | ✅ | ✅ | ✅ | 标签类型 |
| column_names | ✅ | ✅ | ✅ | JSON格式 |
| loss_function | ✅ | ✅ | ✅ | 损失函数 |
| evaluator | ✅ | ✅ | ✅ | 评估器 |
| base_eval_results | ✅ | ✅ | ✅ | 基线评估结果 |
| final_eval_results | ✅ | ✅ | ✅ | 最终评估结果 |
| loss | ✅ | ✅ | ✅ | 训练损失历史 |
| training_evaluator_evaluation | ✅ | ✅ | ✅ | 训练评估历史 |
| create_time | ✅ | ✅ | ✅ | 创建时间 |
| update_time | ✅ | ✅ | ✅ | 更新时间 |

## 关键差异总结

### ❌ 需要修复的主要问题

1. **表名不一致**:
   - 文档需要将 `training_tasks_v2` 改为 `training_tasks`
   - 文档需要将 `dataset_info_v2` 改为 `dataset_info`

2. **training_tasks 表缺失字段**（文档中没有但实际有）:
   - `HF_subset`: HuggingFace数据集子配置
   - `device`: 训练设备配置
   - `updated_at`: 更新时间
   - `loss_data`: 训练损失数据JSON
   - `service_instance_id`: 服务实例ID
   - `process_pid`: 进程PID
   - `process_status`: 进程状态

3. **training_tasks 表多余字段**（文档中有但实际没有）:
   - `duration_seconds`: 持续时间（应该从 started_at 和 completed_at 计算）

## 建议修复方案

### 更新文档的 training_tasks 表结构

```sql
CREATE TABLE training_tasks (
  task_id VARCHAR(64) PRIMARY KEY COMMENT '任务唯一ID',
  task_name VARCHAR(256) COMMENT '任务名称',
  description TEXT COMMENT '任务描述',
  train_type VARCHAR(32) NOT NULL COMMENT '训练类型: embedding, reranker',
  dataset_name_or_path TEXT NOT NULL COMMENT '数据集名称或路径',
  HF_subset VARCHAR(64) COMMENT 'HuggingFace数据集子配置名称',
  output_dir TEXT NOT NULL COMMENT '模型输出路径',
  model_name_or_path TEXT COMMENT '模型名称或路径',
  embedding_dim INT COMMENT '模型维度',
  device VARCHAR(64) COMMENT '训练设备: cpu, cuda:0, cuda:1, auto等',
  status VARCHAR(32) NOT NULL DEFAULT 'PENDING' COMMENT '训练状态',
  progress FLOAT NOT NULL DEFAULT 0.0 COMMENT '训练进度(0.0-100.0)',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  started_at DATETIME COMMENT '开始时间',
  completed_at DATETIME COMMENT '完成时间',
  final_model_path TEXT COMMENT '最终模型路径',
  error_message TEXT COMMENT '错误信息',
  loss_data TEXT COMMENT '训练loss汇总数据JSON',
  training_params TEXT COMMENT '训练参数JSON',
  service_instance_id VARCHAR(128) COMMENT '启动该任务的服务实例ID',
  process_pid INT COMMENT '训练进程PID',
  process_status VARCHAR(32) NOT NULL DEFAULT 'STOPPED' COMMENT '进程状态'
);
```

### 更新文档的表名引用

将所有 `training_tasks_v2` 和 `dataset_info_v2` 的引用改为 `training_tasks` 和 `dataset_info`。