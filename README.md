# Learn and Make SLM

数学や理論的な背景を一旦抜きにして、工学的にTransformerやそのベースとなる基礎的なモジュールを学び、作ってみる。
それらを組み合わせて、最終的に実際に動くSmall Language Modelを作ってみる。

色々書いてある[Scrapbox](https://scrapbox.io/sushichan044-jam/Transformer%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E5%B0%91%E3%81%97%E7%90%86%E8%A7%A3%E3%81%97%E3%81%9F%E6%B0%97%E3%81%AB%E3%81%AA%E3%82%8B%E4%BC%9A)

## 環境構築

uvを使えばいい

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## 実行

### 学習

任意のconfigファイルを指定して学習を行う。
configファイルについては、[config](config)ディレクトリを参照。

```bash
uv run train.py <config_file>
```

各モジュールはPyTestを使い、PyTorch標準のモジュールなどと出力が一致するか簡易的にテストを行っている。

### 推論

任意のconfigファイルを指定して推論を行う。
以下の例以外はhelpを参照。

```bash
# インタラクティブモードで起動
uv run infer.py <config_file>
# temperature及びtop-kを指定して推論(temp=1.0, top-k=10)
uv run infer.py <config_file> --temperature 1.0 --top-k 10
# プロンプトを指定して推論(インタラクティブモードではない)
uv run infer.py <config_file> --prompt "Hello,"
```

## Tensor の型付け

可読性の観点から、jaxtyping を使って Tensor に型付けすることを推奨する。

<https://docs.kidger.site/jaxtyping/api/array/>

学習でのオーバーヘッドを避けるため `nn.Module` の実装に `@jaxtyped` を付与しないことを推奨しており、代わりに pytest を実行すると自動で jaxtyping の runtime checking が実行されるようにしてある。

そのため、各 Module ごとに、出力の shape を見る程度の簡単なテストを追加して CI で Tensor の型付けが正しいか確認可能にすることを推奨する。 ([tests/models/transformer/decoder_layer_test.py](tests/models/transformer/decoder_layer_test.py)程度のもので良い)
