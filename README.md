# What Is This
microsoft社製のvottでラベル付けを行ったアノテーションをyolo v7以降の転移学習用のデータに変換するスクリプト

# How To Use
現在サポートしているバージョンは YOLOv7 のみですので v8 を利用する方はファイルを各自設定してご利用ください。
## 変換方法
### 事前準備
まず以下のコマンドを実行して必要なモジュールをインストールします。<br>
```sh
pip install -r req.txt
```

### 基本的な使い方
```sh
python main.py ./test-data/vott-json-export
```
このように実行すると`./test-data/vott-json-export`にあるVoTTのエクスポートデータをデフォルトディレクトリ`./output`へバリデーションデータを元データから30%分引き出して出力します。<br>

### 出力先変更
```sh
python main.py ./test-data/vott-json-export -o ./out
```
このようにすると基本的な使い方と同じ内容が`./out`へ出力されるようになります。<br>

### バリデーションデータの参照設定
```sh
python main.py ./test-data/vott-json-export -v 0.5
```
このようにすると元データから50%の確率でバリデーションデータを参照して出力します。<br>
0-1の小数点で設定可能です。(0に設定した場合以下の`false`と同様の動作をします)<br>
```sh
python main.py ./test-data/vott-json-export -v false
```
このようにするとバリデーションデータ元データから参照しないようにできます。<br>
個別でバリデーションデータを用意したい場合はこのようにしてください。<br>

### サフィックス設定
```sh
python main.py ./test-data/vott-json-export -s test-1
```
このように設定すると、出力ファイルおよびYOLOでの転移学習時の出力ファイル名へ名前を追加することができます。<br>
同じデータを数値を変えて保存したい場合に重複しないようにすることが可能です。<br>
読み込んだVoTTのプロジェクト名が`test-set`の場合`test-set-test-1`といった感じに追加されます。<br>


### 引数リスト
```
input
--output -o
--valPercent -v (0-1 の小数点で)
--suffix -s
```

## YOLOv7 での使い方
出力されたフォルダを(YOLOv7の場合)YOLOのrootディレクトリ内(`train.py`と同じ階層)にある`data`フォルダ内に移動します。<br>
その後`yolov7-train-start.sh`(Linux)および`yolov7-train-start.bat`(Windows)を実行すればすぐに転移学習が可能です。<br>
それぞれのスクリプトは編集可能ですので各自で設定して利用してください。<br>