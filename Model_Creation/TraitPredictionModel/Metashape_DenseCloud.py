import Metashape

# 1. Metashapeプロジェクトファイル（.psx）を開く
doc = Metashape.app.document
doc.open(r"Y:\0nodera\03_2025_Kawatabi_SORA_Soy\03_Metashape\0606\696photos.psx")  # フルパスを指定

# 2. アクティブチャンクを取得
chunk = doc.chunk

# 3. 点群生成のための設定
# クオリティは High / Medium / Low / Ultra / Lowest のいずれか
# Filteringは Aggressive / Moderate / Mild / Disabled
chunk.buildDepthMaps(quality=Metashape.HighQuality, filtering=Metashape.ModerateFiltering)
chunk.buildDenseCloud()

# 4. 保存（必要なら）
doc.save()
