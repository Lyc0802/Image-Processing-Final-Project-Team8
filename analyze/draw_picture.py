from PIL import Image
import os

def make_grid(input_dir, output_path="output.png", grid_size=4):
    # 取得目錄內前 16 張圖片
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[: grid_size * grid_size]

    if len(files) < grid_size * grid_size:
        raise ValueError("圖片不足 16 張")

    # 讀取所有圖片
    imgs = [Image.open(os.path.join(input_dir, f)) for f in files][:16]

    # 統一圖片大小（以第一張為基準）
    w, h = imgs[0].size
    resized_imgs = [img.resize((w, h)) for img in imgs]

    # 建立大圖
    grid_w = w * grid_size
    grid_h = h * grid_size
    new_img = Image.new("RGB", (grid_w, grid_h))

    # 逐格貼上
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            new_img.paste(resized_imgs[idx], (col * w, row * h))
            idx += 1

    # 儲存結果
    new_img.save(output_path)
    print(f"完成！已輸出 {output_path}")

# 使用方式
make_grid("display_result_2")
