# AI Mosaic Noise Remover

画像の解像度（縦横のピクセル数）を変更せず、モザイク状のノイズだけを除去してディテールを復元するデスクトップアプリケーションです。

## 特徴

- **解像度固定**: 入力画像と同じ解像度で出力（`outscale=1`）
- **Real-ESRGAN**: 高品質な超解像モデルを使ったノイズ除去
- **復元強度スライダー**: ユーザーがデノイズの強さを 0.0〜1.0 で調整可能
- **リアルタイム比較**: 元画像と処理後画像をボタン一つで切り替え
- **非同期処理**: `threading` により処理中も GUI がフリーズしない
- **ドラッグ＆ドロップ**: TkinterDnD2 対応環境では画像ファイルを D&D で入力可能
- **CUDA 対応**: GPU が利用可能な場合は自動的に CUDA で推論

## セットアップ

### 前提条件

- Python 3.9 以上
- (推奨) NVIDIA GPU + CUDA Toolkit

### インストール

```bash
# PyTorch (CUDA 付き) を公式の手順でインストール
# https://pytorch.org/get-started/locally/
# 例: CUDA 11.8 の場合
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# その他の依存ライブラリをインストール
pip install -r requirements.txt
```

もしくは一括で:

```bash
pip install customtkinter tkinterdnd2 Pillow numpy torch torchvision \
            realesrgan basicsr gfpgan opencv-python
```

### モデルの重み

初回実行時に `RealESRGAN_x4plus.pth` が自動ダウンロードされ、`weights/` ディレクトリに保存されます。手動で配置する場合は以下から取得してください:

- <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth>

## 使い方

```bash
python app.py
```

1. **Open Image…** ボタンまたはドラッグ＆ドロップで画像を読み込み
2. サイドバーの **Denoise Strength** スライダーで復元の強さを設定
3. **▶ Process** ボタンで AI デノイズを実行
4. **Toggle Original / Result** ボタンで元画像と結果を比較
5. **Save Result…** ボタンで結果を保存

## プロジェクト構成

```
dreamcometrue/
├── app.py              # メインアプリケーション
├── requirements.txt    # Python 依存ライブラリ
├── weights/            # モデルの重み (自動ダウンロード)
└── README.md
```

## 技術スタック

| カテゴリ | ライブラリ |
|---------|-----------|
| GUI | customtkinter |
| 画像処理 | Real-ESRGAN (basicsr, realesrgan) |
| 推論エンジン | PyTorch (CUDA 優先) |
| 画像 I/O | OpenCV, Pillow |
| ドラッグ＆ドロップ | TkinterDnD2 |

## ライセンス

MIT