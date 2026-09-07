# Dynamic Director AI - 実行手順ガイド

このドキュメントでは、現在実装されているPoC（概念実証）のコードをGitHubにアップロードし、自動ビルド(CI)を回してAndroid端末にインストール・実行するまでの手順を説明します。

## 1. GitHubへのプッシュと自動ビルド

このプロジェクトには、GitHub Actionsを用いた自動ビルド（CI）が設定されています（`.github/workflows/android-build.yml`）。GitHubにコードをプッシュするだけで、自動的にAndroidアプリのAPKファイルが生成されます。

### 手順
1. GitHub上で新しい空のリポジトリ（例: `dynamic-director-poc`）を作成します。
2. ターミナル（コマンドプロンプトやPowerShell等）を開き、本プロジェクトのルートディレクトリ（`dynamic-director-poc`）に移動します。
3. 以下のコマンドを順に実行して、コードをGitHubにプッシュします。

```bash
# gitリポジトリの初期化
git init

# すべてのファイルをステージングしてコミット
git add .
git commit -m "Initial commit for Dynamic Director PoC (Phase 1-3)"

# デフォルトブランチをmainに設定
git branch -M main

# リモートリポジトリ（作成したGitHubリポジトリのURL）を登録
# ※ 以下のURLはご自身のリポジトリURLに置き換えてください
git remote add origin https://github.com/あなたのユーザー名/リポジトリ名.git

# GitHubへコードをプッシュ
git push -u origin main
```

4. プッシュ完了後、GitHubリポジトリのページを開き、上部の「**Actions**」タブをクリックします。
5. 「Android CI」というワークフローが自動的に実行されているはずですので、完了（緑色のチェックマーク）するまで数分待ちます。
6. 完了したビルドをクリックして詳細ページを開くと、ページ下部の「**Artifacts**」セクションに `app-debug` （ビルド済みのAPKファイル群）が生成されています。これをクリックしてダウンロードします。

---

## 2. Android端末へのインストール

GitHub Actionsで生成されたAPKをAndroid端末にインストールします。

### 方法A: 端末のブラウザから直接ダウンロードする場合
1. Android端末のブラウザからご自身のGitHubリポジトリのActionsページにアクセスし、Artifactの `app-debug.zip` をダウンロードします。
2. ダウンロードしたZipファイルを端末内で解凍し、中にある `app-debug.apk` をタップして開きます。
3. （初回の場合）「提供元のわからないアプリのインストール」を許可するよう求められるため、ブラウザやファイル管理アプリの設定からインストールを許可します。
4. インストールボタンを押して完了させます。

### 方法B: PC経由・adbコマンドを使用する場合
PCに Android SDK Platform-Tools（`adb`）がインストールされており、端末のUSBデバッグが有効な場合はこちらが簡単です。
1. GitHubからPCに `app-debug.zip` をダウンロードし、展開して `app-debug.apk` を取り出します。
2. Android端末をUSBケーブルでPCに接続し、以下のコマンドを実行します。
```bash
adb install app-debug.apk
```

---

## 3. アプリの実行と動作確認

1. Android端末のホーム画面から、「**Dynamic Director PoC**」アプリを起動します。
2. 起動直後に「写真と動画の撮影をアプリに許可しますか？」という**カメラの権限リクエスト**が表示されますので、「許可」を選択してください。
3. 許可されると、スマートフォンのカメラ映像（プレビュー）が画面全体に表示されます。
4. 画面左上に、開発者用のダッシュボードとして「`Score: --%`」「`Waiting for Detection...`」というテキストが緑色でオーバーレイ表示されていることを確認します。

> [!NOTE]
> 現在の実装（フェーズ3まで）では、カメラ映像の取得とUI表示の基盤までが完成しています。人物の自動検出や、設定値に基づくスコアリング、および自動シャッター機能の実装はこれからのフェーズ（Phase 4, 5）で行われます。
