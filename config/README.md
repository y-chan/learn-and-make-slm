# Configの説明

configファイルはYAML形式で記述されており、以下のような形式である。

configファイルに内容を追加する場合は、[config.py](../config.py)を編集してください。

```yaml
# データセット名、現状はSimpleStories-Both(日本語と英語の両方)、SimpleStories-JA、SimpleStoriesに対応
dataset: "SimpleStories-Both"
# トークナイザー名、tiktokenで指定可能なもの
tokenizer: "cl100k_base"
path:
  # ログディレクトリ、自由に指定可能
  log_dir: "logs/simple_stories_gpt_oss"
model:
  # モデルタイプ、現状はgpt-oss、gpt-2のみ対応
  model_type: "gpt-oss"
  # モデル全体の隠れ層の次元
  d_model: 512
  # Attentionのヘッド数
  n_heads: 8
  # Grouped Query Attentionのグループ数、gpt-ossの際は必須
  n_groups: 2
  # TransformerのDecoderの層数
  n_layers: 16
  # RoPEのスケールファクター、指定しなければデフォルト値1でRoPEを使用、それ以上の場合はYaRNに切り替わる
  rope_scale_factor: 1.0
  # Gated Attentionの有効/無効、指定しなければ無効
  use_sigmoid_gate: false

train:
  # 乱数シード
  manual_seed: 1234
  # 入力の最大長
  max_length: 1024
  # エポック数
  epochs: 1
  # バッチサイズ、この設定でRTX3090(VRAM 24GB)で余裕を持って学習可能
  batch_size: 4
  # 学習率
  learning_rate: 0.0005
  # AdamWのモーメンタムパラメータ
  betas: [0.9, 0.999]
  # L2正則化の強さ
  weight_decay: 0.1
  # 学習率の減衰率
  lr_decay: 0.995
  # チェックポイントの保存間隔(エポック数)
  save_epochs: 1
  # ログの出力間隔、勾配蓄積のステップ数x指定した間隔で出力
  logging_steps: 1
  # 勾配蓄積のステップ数、擬似的にバッチサイズを大きくする
  gradient_accumulation_steps: 16
```
