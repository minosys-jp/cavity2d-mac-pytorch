# cavity2d-mac-pytorch
Analyze Cavity 2D problem by MAC method with PyTorch

二次元キャビティ問題を python と MAC を使って解くサンプルです。
数値ライブラリとして、PyTorch を使用しています。（もちろん PyTorch ライブラリをインストールしておく必要があります。）


# 初期条件
内部で p=0, u=v=0

# 境界条件
y = 1 で u=u0
その他の条件は[記事](https://minosys.com/wp/2023/01/20/62/)の通り。

# フォルダの説明

## common

CPU/GPU/Triton で共通に使用するクラスを定義しています。境界条件はこの中に記述されています。

## cpu

PyTorch の CPU コードを使った MAC 法のコードです。

## gpu

PyTorch の GPU コードを使った MAC 法のコードです。データの受け渡し以外、CPU とほぼ変わりません。

## gpu_triton

PyTorch にバンドルされている[Triton](https://triton-lang.org/main/index.html) を使って高速化したものです。

## measure.py

以下の PC 環境で、3つの解法の速度を比べ、グラフ化するプログラムが格納されています。CPU, GPU, Triton の各パッケージを呼び出して使用しています。
利用には matplotlib が必要です。

|パーツ|内容|
|---|---|
|CPU|Intel Core i7 13700K|
|GPU|NVIDIA RTX 4090 GDDR6 24GB|
|RAM|DDR4 128GB|
|SSD|WD_BLACK SN7100 4TB|
