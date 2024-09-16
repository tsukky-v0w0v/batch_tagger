# batch_tagger
ローカル環境でSmilingWolf氏のonnxモデルを動かす<br>
https://huggingface.co/SmilingWolf

## インストール
```cmd
git clone https://github.com/tsukky-v0w0v/batch_tagger.git
cd batch_tagger
python -m venv vevn
.\venv\scripts\activate
pip install -r requirements.txt
```
## GPUを使用する
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements<br>

CUDA12.6.1 cuDNN9.4.0で動作確認しています。
```cmd
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```
CUDA11.xの場合はcuDNN8.xにて
```cmd
pip install onnxruntime-gpu
```
でGPU使用ができる、はず。(未確認)

## 使い方
`run.py`を実行します。

### 引数解説
`--target <パス>`
- 画像ファイル、もしくはフォルダを指定します。

`--use_rating`
- レーティングタグを使用します。
- general, sensitive, questionable, explicitの4つから最もスコアが高いものが選出されます。

`--use_character`
- キャラクタータグを使用します。

`--character_threshold <数値>`
- キャラクタータグの閾値です。
- 未指定時のデフォルトは0.85です。

`--use_general`
- 一般タグを使用します。

`--general_threshold <数値>`
- 一般タグの閾値です。
- 未指定時のデフォルトは0.35です。

`--use_recommended_threshold`
- 後述の`models.py`に記載された閾値を使用します。
- 対象は一般タグのみです。
- `--general_threshold`の値は無視されます。

`--ext <拡張子>`
- タグファイルの拡張子です。
- 未指定時のデフォルトは`.txt`です。

`--overwrite`
- タグファイルを上書きします。

`--weighted_captions`
- スコアをタグの重みとして記載します。
- 例: (1girl:0.81123), (blach hair:0.75332), ...
- sdxlではsd-scriptsがweighted_captionsに対応していないので使いどころはないかも

`--recursive`
- 対象にサブフォルダを含めます。

`--additional_tag <タグ>`
- 追加タグです。
- 複数のタグを追加する場合、`,`で区切るか、`--additional_tag`引数を複数回使用できます。
- 例: 1girl, black hairを追加する場合<br>
  パターン1: --additional_tag "1girl, black hair"<br>
  パターン2: --additional_tag "1girl" --additional_tag "black hair"

`--exclude_tag <タグ>`
- 除外タグです。
- 複数タグを除外する場合、使い方はadditional_tagと同様です。

`--all_sort`
- タグ全体をスコアの降順にソートします。
- 使用しない場合、`additional_tag`、`rating_tag`、`character_tag`、`general_tag`の順に記載されます。

`--cpu`
- CPUのみを使用します。
- GPUを使用したい場合は後述。

`--model`
- 後述の`models.py`に記載したモデル名を指定します。
- デフォルトは`wd-v1-4-swinv2-tagger-v2`です。

`--batch_size <数値>`
- バッチサイズです。
- デフォルトは10です。

`config_file <パス>`
- 上記引数をtomlファイルで記述できます。
- 例: 
  ```toml
    use_rating = true
    use_character = true
    use_general = true

    character_threshold = 0.85
    general_threshold = 0.35
    use_recommended_threshold = false

    model = "wd-eva02-large-tagger-v3"
    batch_size = 10

    overwrite = true
    weighted_captions = false
    recursive = true
    additional_tag = []
    exclude_tag = ["1girl, black hair"]
    all_sort = false
  ```

### models.py
モデル情報のdictを記載しているファイルです。
```python
models = {
    "wd-swinv2-tagger-v3": {
        "repo_id": "SmilingWolf/wd-swinv2-tagger-v3", 
        "threshold": 0.2653
    },
    "wd-convnext-tagger-v3": {
        "repo_id": "SmilingWolf/wd-convnext-tagger-v3", 
        "threshold": 0.2682
    }, 
}
```
modelsのキーには`--model`で指定するモデル名を記載します。<br>
値にはhugging faceのリポジトリと、`--use_recommended_threshold`指定時に使用する閾値を記載します。
現在、hugging faceの公式リポジトリに記載がある閾値を入力しています。<br>
モデルを追加する場合はここに追記していってください。

### バッチファイル
```batch
@echo off
chcp 65001

REM 設定ファイル
set config_file="config.toml"

REM Python仮想環境を有効化
set venv_path="%~dp0\venv\Scripts\activate.bat"
call "%venv_path%"

REM スクリプト実行
set script="%~dp0\run.py"

REM コマンドライン引数を渡す
python %script% ^
    --target "%~1" ^
    --config_file %config_file%

pause
```
こんな感じのバッチファイルを作っておくと便利かもしれない

## 参考
- https://github.com/picobyte/stable-diffusion-webui-wd14-tagger
- https://github.com/corkborg/wd14-tagger-standalone