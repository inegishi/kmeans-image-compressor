# 🎨 K-Means Image Compressor

This project uses the **K-means clustering algorithm** to compress an image by reducing the number of unique colors. The result is a visually similar image that uses far fewer RGB values, which can help with file size reduction and simplify image data for machine learning tasks.

---

## 📌 How It Works

- Loads and normalizes an image
- Reshapes the image into a 2D array of pixels (each with 3 RGB values)
- Runs **K-means clustering** to group similar colors into `K` clusters
- Replaces each pixel with the nearest cluster centroid color
- Reshapes the result back into an image and displays the compressed version

---

## 🖼️ Example
![image](https://github.com/user-attachments/assets/59780c1d-2948-4727-a69f-375468493ada)

